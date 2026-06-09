# Multimodal Deep Learning for Peptide Activity Prediction

Research code for the project **"Multimodal Deep Learning for Modified Peptide Design"**, conducted as part of a visiting research collaboration with the Machine Learning Department at Carnegie Mellon University (Supervisor: Prof. Barnabas Poczos).

## Overview

This repository implements a multimodal deep learning pipeline for predicting the biological activity of peptide sequences across four functional categories: antimicrobial (AMP), antioxidant (AOP), cell-penetrating (CPP), and antihypertensive (AHP).

The core contribution is a **Transformer-based fusion model** that combines two complementary feature modalities:
- **Protein language model embeddings** from ESM-2, extracted with learned attention pooling
- **Physicochemical features** encoding composition, charge, hydrophobic moment, Boman index, functional motifs, and estimated pI

These modalities are projected to a shared space and fused via a Transformer encoder, enabling cross-modal interaction before a multi-label classification head. A Random Forest baseline using the same concatenated features is trained in parallel for comparison.

## Motivation

Standard approaches to peptide activity prediction typically rely on either sequence-only models or hand-crafted biochemical descriptors. This project explores whether combining both — using a protein language model to capture evolutionary and structural context alongside domain-informed physicochemical features — improves predictive performance and biological interpretability under realistic class-imbalance conditions.

## Repository Structure

```
peptide-ml-research/
├── train.py               # End-to-end training and comparison script
├── requirements.txt
├── src/
│   ├── embeddings.py      # ESM-2 embedder with attention pooling
│   ├── features.py        # Physicochemical feature extraction (26-dim)
│   ├── models.py          # TransformerFusion + RandomForest baseline
│   └── evaluate.py        # Per-label metrics, ROC curves, comparison plots
└── data/
    └── processed/         # Place training CSV here (see Data section)
```

## Method

### Attention Pooling

Rather than mean-pooling all ESM-2 token embeddings, a lightweight learned attention layer assigns position-specific importance weights before aggregation. This allows the model to down-weight structurally uninformative regions and focus on activity-relevant residues.

### Transformer Fusion

Each modality (ESM embedding and physicochemical vector) is independently projected to a 256-dimensional space and treated as one token in a 2-token sequence. A 2-layer Transformer encoder with pre-layer normalisation learns cross-modal interactions, and the fused output is mean-pooled before passing through an MLP head producing multi-label sigmoid outputs.

### Baseline

A Random Forest classifier is trained per label on the concatenated feature vector (ESM + physicochemical, 1306 dimensions total), with `balanced_subsample` class weighting to handle label imbalance.

## Data

The pipeline expects a CSV at `data/processed/training_data.csv` with columns:

| Column     | Description                          |
|------------|--------------------------------------|
| `sequence` | Amino acid sequence (single-letter)  |
| `length`   | Sequence length                      |
| `AMP`      | Binary label (antimicrobial)         |
| `AOP`      | Binary label (antioxidant)           |
| `CPP`      | Binary label (cell-penetrating)      |
| `AHP`      | Binary label (antihypertensive)      |

Peptides are filtered to lengths 10–30 aa. A stratified sub-sample of up to 5,000 sequences is used for training by default.

## Usage

```bash
pip install -r requirements.txt

# Train with ESM-2-650M (default, ~3 GB VRAM)
python train.py --data data/processed/training_data.csv

# Train with ESM-2-3B (higher capacity, ~24 GB VRAM)
python train.py --data data/processed/training_data.csv --model-size 3B --epochs 100
```

Output files:
- `esm_embeddings.npy` — cached ESM embeddings
- `phys_features.npy` — physicochemical feature matrix
- `rf_model.pkl` — trained Random Forest
- `transformer_fusion.pt` — TransformerFusion weights
- `model_comparison.png` — per-label F1 / AUC bar chart + ROC curves

## Evaluation

The comparison script reports per-label and macro-averaged F1, AUC-ROC, and average precision for both models on a held-out 20% validation split. Example output:

```
=== Model Comparison (validation set) ===
Label    RF-F1   TF-F1  RF-AUC  TF-AUC   RF-AP   TF-AP
-------------------------------------------------------
AMP      0.741   0.783   0.821   0.856   0.764   0.801
AOP      0.612   0.658   0.743   0.779   0.631   0.672
CPP      0.583   0.621   0.712   0.748   0.598   0.639
AHP      0.527   0.574   0.681   0.714   0.541   0.581
-------------------------------------------------------
macro    0.616   0.659   0.739   0.774   0.634   0.673
```

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- fair-esm 2.0+
- scikit-learn 1.3+
- pandas, numpy, matplotlib, tqdm

## Acknowledgements

This work is part of a collaborative AI-for-Science project at the Machine Learning Department, Carnegie Mellon University. Protein language model embeddings are provided by Meta's [ESM-2](https://github.com/facebookresearch/esm) (MIT License).
