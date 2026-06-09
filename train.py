"""
Training script: RF baseline vs TransformerFusion on peptide activity prediction.

Workflow:
  1. Load and filter peptide dataset.
  2. Extract ESM-2 embeddings (with attention pooling).
  3. Extract physicochemical features.
  4. Train Random Forest baseline on concatenated features.
  5. Train TransformerFusion on separate modalities.
  6. Compare both models on held-out validation set.
  7. Save models and comparison figure.

Usage:
    python train.py --data data/processed/training_data.csv
    python train.py --data data/processed/training_data.csv --model-size 3B --epochs 100
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch

from src.embeddings import ESMEmbedder
from src.features import extract_batch
from src.models import FUNCTION_LABELS, RandomForestPredictor, TransformerTrainer
from src.evaluate import compare_models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train peptide activity prediction models.")
    p.add_argument("--data",        default="data/processed/training_data.csv",
                   help="Path to CSV with columns: sequence, AMP, AOP, CPP, AHP, length")
    p.add_argument("--model-size",  default="650M", choices=["650M", "3B"],
                   help="ESM-2 model variant")
    p.add_argument("--batch-size",  type=int, default=32,
                   help="Batch size for ESM embedding extraction")
    p.add_argument("--epochs",      type=int, default=50,
                   help="Training epochs for TransformerFusion")
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Cap on training sequences (stratified sample)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_data(path: str, max_samples: int) -> tuple:
    df = pd.read_csv(path)
    df = df[(df['length'] >= 10) & (df['length'] <= 30)].copy()

    # Stratified sub-sample to respect class balance
    if len(df) > max_samples:
        per_class = max_samples // len(FUNCTION_LABELS)
        frames = []
        covered = set()
        for fn in FUNCTION_LABELS:
            pos = df[df[fn] == 1]
            n = min(len(pos), per_class)
            chosen = pos.sample(n, random_state=42)
            frames.append(chosen)
            covered.update(chosen.index)
        remaining = df.drop(index=list(covered))
        extra = max_samples - sum(len(f) for f in frames)
        if extra > 0 and len(remaining) > 0:
            frames.append(remaining.sample(min(extra, len(remaining)), random_state=42))
        df = pd.concat(frames).drop_duplicates()

    sequences = df['sequence'].tolist()
    y = df[FUNCTION_LABELS].values.astype(np.float32)
    return sequences, y


def main() -> None:
    args = parse_args()

    # 1. Load data
    print(f"Loading data from {args.data} ...")
    sequences, y = load_data(args.data, args.max_samples)
    print(f"  {len(sequences)} peptides after filtering")
    for i, fn in enumerate(FUNCTION_LABELS):
        print(f"  {fn}: {int(y[:, i].sum())} positives ({y[:, i].mean()*100:.1f}%)")

    # 2. ESM embeddings (attention pooling)
    print(f"\nExtracting ESM-2-{args.model_size} embeddings with attention pooling ...")
    embedder = ESMEmbedder(model_size=args.model_size, batch_size=args.batch_size)
    esm_X = embedder.encode(sequences)                   # (N, 1280)
    np.save("esm_embeddings.npy", esm_X)
    print(f"  Saved → esm_embeddings.npy  shape={esm_X.shape}")

    # 3. Physicochemical features
    print("Extracting physicochemical features ...")
    phys_X = extract_batch(sequences)                    # (N, 26)
    np.save("phys_features.npy", phys_X)
    print(f"  Saved → phys_features.npy  shape={phys_X.shape}")

    # 4. Concatenated feature matrix for RF baseline
    X = np.hstack([esm_X, phys_X])                      # (N, 1306)

    # 5. Train RF baseline
    print("\n--- Random Forest Baseline ---")
    rf = RandomForestPredictor()
    rf.fit(X, y)
    print("  Validation metrics:")
    for fn, m in rf.val_metrics.items():
        print(f"    {fn}: F1={m['f1']:.3f}  AUC={m['auc']:.3f}")

    # 6. Train TransformerFusion
    print("\n--- TransformerFusion ---")
    trainer = TransformerTrainer(
        esm_dim=esm_X.shape[1],
        phys_dim=phys_X.shape[1],
        device=args.device,
        epochs=args.epochs,
    )
    trainer.fit(esm_X, phys_X, y)
    print(f"  Final macro-F1 (val): {trainer.val_metrics.get('macro_f1', 0):.3f}")

    # 7. Compare on common held-out split
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    _, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    rf_proba = rf.predict_proba(X[val_idx])
    tf_proba = trainer.predict_proba(esm_X[val_idx], phys_X[val_idx])
    compare_models(y[val_idx], rf_proba, tf_proba, save_path="model_comparison.png")

    # 8. Save models
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    torch.save(trainer.model.state_dict(), "transformer_fusion.pt")
    print("\nModels saved: rf_model.pkl, transformer_fusion.pt")


if __name__ == "__main__":
    main()
