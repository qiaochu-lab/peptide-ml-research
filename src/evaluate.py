"""
Evaluation utilities: per-label metrics and RF vs TransformerFusion comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from typing import Dict

FUNCTION_LABELS = ['AMP', 'AOP', 'CPP', 'AHP']


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, dict]:
    """
    Compute per-label F1, AUC-ROC, and average precision.

    Args:
        y_true:  (N, 4) binary ground-truth labels
        y_proba: (N, 4) predicted probabilities in [0, 1]
    Returns:
        dict mapping label name → {f1, auc, ap}; plus 'macro' averages.
    """
    y_pred = (y_proba > 0.5).astype(int)
    results = {}

    for i, label in enumerate(FUNCTION_LABELS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ypr = y_proba[:, i]
        has_pos = yt.sum() > 0
        results[label] = {
            'f1':  f1_score(yt, yp, zero_division=0),
            'auc': roc_auc_score(yt, ypr) if has_pos else 0.0,
            'ap':  average_precision_score(yt, ypr) if has_pos else 0.0,
        }

    results['macro'] = {k: np.mean([results[l][k] for l in FUNCTION_LABELS]) for k in ('f1', 'auc', 'ap')}
    return results


def compare_models(
    y_true:    np.ndarray,
    rf_proba:  np.ndarray,
    tf_proba:  np.ndarray,
    save_path: str = "model_comparison.png",
) -> None:
    """
    Side-by-side bar chart comparing RF baseline vs TransformerFusion
    on per-label F1 and AUC-ROC, plus ROC curves for each label.

    Saves figure to save_path and prints a summary table.
    """
    rf_m = compute_metrics(y_true, rf_proba)
    tf_m = compute_metrics(y_true, tf_proba)

    labels = FUNCTION_LABELS
    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Bar charts: F1 and AUC ---
    for ax, metric in zip(axes[0], ('f1', 'auc')):
        rf_vals = [rf_m[l][metric] for l in labels]
        tf_vals = [tf_m[l][metric] for l in labels]
        ax.bar(x - w / 2, rf_vals, w, label='Random Forest (baseline)', color='steelblue', alpha=0.85)
        ax.bar(x + w / 2, tf_vals, w, label='TransformerFusion',        color='coral',     alpha=0.85)
        for xi, (rv, tv) in enumerate(zip(rf_vals, tf_vals)):
            ax.text(xi - w / 2, rv + 0.01, f'{rv:.2f}', ha='center', fontsize=8)
            ax.text(xi + w / 2, tv + 0.01, f'{tv:.2f}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Per-label {metric.upper()}: RF vs TransformerFusion')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # --- ROC curves per label ---
    for ax, (i, label) in zip(axes[1], enumerate(labels)):
        RocCurveDisplay.from_predictions(
            y_true[:, i], rf_proba[:, i],
            name='Random Forest', ax=ax, color='steelblue',
        )
        RocCurveDisplay.from_predictions(
            y_true[:, i], tf_proba[:, i],
            name='TransformerFusion', ax=ax, color='coral',
        )
        ax.set_title(f'ROC — {label}')
        ax.legend(fontsize=8)

    # handle 4th subplot if only 4 labels
    if len(labels) < 4:
        axes[1, -1].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved → {save_path}")

    # Summary table
    print("\n=== Model Comparison (validation set) ===")
    print(f"{'Label':<8} {'RF-F1':>7} {'TF-F1':>7} {'RF-AUC':>8} {'TF-AUC':>8} {'RF-AP':>7} {'TF-AP':>7}")
    print("-" * 56)
    for l in labels:
        print(
            f"{l:<8} {rf_m[l]['f1']:>7.3f} {tf_m[l]['f1']:>7.3f} "
            f"{rf_m[l]['auc']:>8.3f} {tf_m[l]['auc']:>8.3f} "
            f"{rf_m[l]['ap']:>7.3f} {tf_m[l]['ap']:>7.3f}"
        )
    print("-" * 56)
    l = 'macro'
    print(
        f"{'macro':<8} {rf_m[l]['f1']:>7.3f} {tf_m[l]['f1']:>7.3f} "
        f"{rf_m[l]['auc']:>8.3f} {tf_m[l]['auc']:>8.3f} "
        f"{rf_m[l]['ap']:>7.3f} {tf_m[l]['ap']:>7.3f}"
    )
