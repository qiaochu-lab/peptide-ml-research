"""
Peptide activity prediction models.

Two approaches compared side-by-side:

  RandomForestPredictor
    Classical ML baseline. One independent RF per peptide function label.
    Input: concatenated ESM embeddings + physicochemical features.

  TransformerFusion  (main contribution)
    Multimodal deep learning model. Treats ESM embeddings and physicochemical
    features as two separate modalities, projects each to a shared space, then
    applies a Transformer encoder for cross-modal interaction before a
    multi-label classification head.

    Architecture sketch:
      ESM emb  (1280) ──┐
                         ├── project to (proj_dim) each
      Phys feat (26)  ──┘
             │
      [token_esm, token_phys]  shape (B, 2, proj_dim)
             │
      TransformerEncoder  (n_layers, n_heads)
             │
      mean pool → MLP head → sigmoid → (B, 4) probabilities
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from typing import Dict, Optional, Tuple

FUNCTION_LABELS = ['AMP', 'AOP', 'CPP', 'AHP']


# ---------------------------------------------------------------------------
# Classical baseline
# ---------------------------------------------------------------------------

class RandomForestPredictor:
    """
    Multi-label Random Forest predictor.

    Trains one RF classifier per label using the full concatenated feature
    vector (ESM embedding + physicochemical features) as input.
    Class imbalance is handled via 'balanced_subsample' weighting.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.clfs: Dict[str, RandomForestClassifier] = {
            fn: RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                random_state=random_state,
                n_jobs=-1,
            )
            for fn in FUNCTION_LABELS
        }
        self.val_metrics: Dict[str, dict] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X: (N, D)  concatenated feature matrix
            y: (N, 4)  binary multi-label targets
        """
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        for i, fn in enumerate(FUNCTION_LABELS):
            self.clfs[fn].fit(X_tr, y_tr[:, i])
            preds = self.clfs[fn].predict(X_val)
            proba = self.clfs[fn].predict_proba(X_val)[:, 1]
            self.val_metrics[fn] = {
                'f1':  f1_score(y_val[:, i], preds, zero_division=0),
                'auc': roc_auc_score(y_val[:, i], proba) if y_val[:, i].sum() > 0 else 0.0,
            }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, 4) probability matrix."""
        return np.stack(
            [self.clfs[fn].predict_proba(X)[:, 1] for fn in FUNCTION_LABELS],
            axis=1,
        )


# ---------------------------------------------------------------------------
# Deep learning: Transformer-based multimodal fusion
# ---------------------------------------------------------------------------

class TransformerFusion(nn.Module):
    """
    Cross-modal Transformer for peptide multi-label activity prediction.

    Each modality (ESM language model embedding, physicochemical features)
    is projected to a common dimension and treated as one token in a short
    2-token sequence. A Transformer encoder with pre-LN (for training
    stability) learns cross-modal interactions, and the fused representation
    is mean-pooled and passed through an MLP classification head.
    """

    def __init__(
        self,
        esm_dim:   int = 1280,
        phys_dim:  int = 26,
        proj_dim:  int = 256,
        n_heads:   int = 4,
        n_layers:  int = 2,
        dropout:   float = 0.1,
        n_labels:  int = 4,
    ):
        super().__init__()

        # Independent modality projections
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Transformer encoder (pre-LN for stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Multi-label classification head
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, n_labels),
        )

    def forward(
        self,
        esm_emb:   torch.Tensor,   # (B, esm_dim)
        phys_feat: torch.Tensor,   # (B, phys_dim)
    ) -> torch.Tensor:
        """Returns (B, n_labels) raw logits (apply sigmoid for probabilities)."""
        e = self.esm_proj(esm_emb).unsqueeze(1)    # (B, 1, proj_dim)
        p = self.phys_proj(phys_feat).unsqueeze(1)  # (B, 1, proj_dim)

        tokens = torch.cat([e, p], dim=1)            # (B, 2, proj_dim)
        fused  = self.transformer(tokens)            # (B, 2, proj_dim)
        pooled = fused.mean(dim=1)                   # (B, proj_dim)
        return self.head(pooled)                     # (B, n_labels)


class TransformerTrainer:
    """
    Training wrapper for TransformerFusion.

    Handles batched training with AdamW + cosine LR scheduling,
    gradient clipping, early stopping via best-val-loss checkpoint,
    and final evaluation.
    """

    def __init__(
        self,
        esm_dim:    int = 1280,
        phys_dim:   int = 26,
        device:     str = "cuda" if torch.cuda.is_available() else "cpu",
        lr:         float = 1e-4,
        epochs:     int = 50,
        batch_size: int = 64,
    ):
        self.device     = device
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model      = TransformerFusion(esm_dim=esm_dim, phys_dim=phys_dim).to(device)
        self.optimizer  = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.criterion  = nn.BCEWithLogitsLoss()
        self.val_metrics: Dict[str, float] = {}
        self._best_state: Optional[dict] = None

    def fit(
        self,
        esm_X:  np.ndarray,   # (N, esm_dim)
        phys_X: np.ndarray,   # (N, phys_dim)
        y:      np.ndarray,   # (N, 4) binary labels
    ) -> None:
        idx = np.arange(len(y))
        tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

        esm_t  = torch.tensor(esm_X,  dtype=torch.float32)
        phys_t = torch.tensor(phys_X, dtype=torch.float32)
        y_t    = torch.tensor(y,      dtype=torch.float32)

        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            np.random.shuffle(tr_idx)
            train_loss = 0.0

            for start in range(0, len(tr_idx), self.batch_size):
                batch = tr_idx[start : start + self.batch_size]
                e  = esm_t[batch].to(self.device)
                p  = phys_t[batch].to(self.device)
                yb = y_t[batch].to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(e, p), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(batch)

            self.scheduler.step()
            train_loss /= len(tr_idx)

            if (epoch + 1) % 10 == 0:
                val_loss, val_f1 = self._eval(esm_t[val_idx], phys_t[val_idx], y_t[val_idx])
                print(
                    f"Epoch {epoch+1:3d}/{self.epochs} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f} | macro-F1={val_f1:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        _, self.val_metrics['macro_f1'] = self._eval(
            esm_t[val_idx], phys_t[val_idx], y_t[val_idx]
        )

    @torch.no_grad()
    def _eval(
        self,
        esm_t:  torch.Tensor,
        phys_t: torch.Tensor,
        y_t:    torch.Tensor,
    ) -> Tuple[float, float]:
        self.model.eval()
        logits = []
        for start in range(0, len(y_t), self.batch_size):
            e = esm_t[start : start + self.batch_size].to(self.device)
            p = phys_t[start : start + self.batch_size].to(self.device)
            logits.append(self.model(e, p).cpu())
        logits = torch.cat(logits)
        loss   = self.criterion(logits, y_t).item()
        proba  = torch.sigmoid(logits).numpy()
        preds  = (proba > 0.5).astype(int)
        f1     = f1_score(y_t.numpy(), preds, average='macro', zero_division=0)
        return loss, f1

    @torch.no_grad()
    def predict_proba(self, esm_X: np.ndarray, phys_X: np.ndarray) -> np.ndarray:
        """Returns (N, 4) probability matrix."""
        self.model.eval()
        esm_t  = torch.tensor(esm_X,  dtype=torch.float32)
        phys_t = torch.tensor(phys_X, dtype=torch.float32)
        out = []
        for start in range(0, len(esm_X), self.batch_size):
            e = esm_t[start : start + self.batch_size].to(self.device)
            p = phys_t[start : start + self.batch_size].to(self.device)
            out.append(torch.sigmoid(self.model(e, p)).cpu().numpy())
        return np.vstack(out)
