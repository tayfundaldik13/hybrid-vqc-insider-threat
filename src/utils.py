"""
Utility Functions
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    average_precision_score,
)
import os
import json
from typing import Dict, List


def calculate_metrics(targets, preds, probs) -> Dict[str, float]:
    targets = np.array(targets)
    preds   = np.array(preds)
    probs   = np.array(probs)

    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    has_both = len(np.unique(targets)) > 1

    return {
        "acc":         accuracy_score(targets, preds),
        "kappa":       cohen_kappa_score(targets, preds),
        "mcc":         matthews_corrcoef(targets, preds),
        "precision":   precision_score(targets, preds, zero_division=0),
        "recall":      recall_score(targets, preds, zero_division=0),
        "specificity": specificity,
        "f1":          f1_score(targets, preds, zero_division=0),
        "macro_f1":    f1_score(targets, preds, zero_division=0, average="macro"),
        "auc":         roc_auc_score(targets, probs) if has_both else 0.5,
        "auprc":       average_precision_score(targets, probs) if has_both else 0.0,
        "ece":         calculate_ece(targets, probs),
    }

def find_optimal_threshold(targets, probs, method="f1_max") -> float:
    targets = np.array(targets)
    probs = np.array(probs)

    best_threshold = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.1, 0.9, 50):
        preds = (probs > t).astype(float)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold

def fdr_correction(p_values, alpha=0.05):
    p_arr = np.array(p_values, dtype=float)
    n = len(p_arr)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]
    ranks = np.arange(1, n + 1, dtype=float)

    q_raw = sorted_p * n / ranks
    q_adj = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_adj = np.minimum(q_adj, 1.0)

    q_values = np.empty(n)
    q_values[sorted_idx] = q_adj

    return q_values <= alpha, q_values

def calculate_ece(targets, probs, n_bins=10) -> float:
    targets = np.array(targets)
    probs = np.array(probs)
    total = len(targets)

    if total < 5:
        return float("nan")

    n_bins = min(n_bins, max(2, total // 5))

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if i == n_bins - 1: 
            mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bin_accuracy = targets[mask].mean()
        bin_confidence = probs[mask].mean()
        ece += (n_bin / total) * abs(bin_accuracy - bin_confidence)

    return ece

def plot_reliability_diagram(targets, probs, title, save_path=None, n_bins=10):
    targets = np.array(targets)
    probs = np.array(probs)
    total = len(targets)
    n_bins = min(n_bins, max(2, total // 5))
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(targets[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_counts.append(n_bin)

    ece = calculate_ece(targets, probs, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
    ax1.bar(bin_confs, bin_accs, width=1.0/n_bins, alpha=0.6,
            edgecolor='black', linewidth=0.5, label='Model')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives (Accuracy)')
    ax1.set_title(f'{title}\nECE = {ece:.4f}')
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(probs, bins=n_bins, range=(0, 1), edgecolor='black',
             alpha=0.6, color='steelblue')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Reliability diagram saved: {save_path}")
    plt.close()

    return ece

def plot_confusion_matrix(targets, preds, title, save_path=None):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(targets, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Threat"], yticklabels=["Normal", "Threat"])
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Prediction")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"CM saved: {save_path}")
    plt.close()

def print_stage_report(stage_size, seed, all_fold_metrics,
                       fold_shapiro_ps=None, mann_whitney_p=None):
    print(f"\n{'='*60}")
    print(f"Stage Report: Size={stage_size}, Seed={seed}")
    print(f"{'='*60}")

    for key, values in all_fold_metrics.items():
        print(f"Mean {key.upper():<9}: %{np.mean(values)*100:.2f} (±{np.std(values)*100:.2f})")

    print(f"{'-'*60}")

    if fold_shapiro_ps:
        all_normal = all(p > 0.05 for p in fold_shapiro_ps)
        status = "All Normal" if all_normal else "Some Not Normal"
        print(f"Shapiro (per fold):")
        for i, p in enumerate(fold_shapiro_ps):
            n_str = "successful" if p > 0.05 else "failure"
            print(f"Fold {i+1}: P={p:.4f} {n_str}")
        print(f"→ {status}")

    if mann_whitney_p is not None:
        if mann_whitney_p < 0.001:
            mw_str = "Highly Significant (p<0.001)"
        elif mann_whitney_p < 0.05:
            mw_str = "Significant"
        else:
            mw_str = "Not Significant"
        print(f"Mann-Whitney U: P={mann_whitney_p:.6f} → {mw_str}")

    kappa_scores = all_fold_metrics.get("kappa", [])
    if kappa_scores:
        km = np.mean(kappa_scores)
        if km > 0.8: k_str = "Almost Perfect"
        elif km > 0.6: k_str = "Substantial"
        elif km > 0.4: k_str = "Moderate"
        elif km > 0.2: k_str = "Fair"
        else: k_str = "Slight/Poor"
        print(f"Cohen's Kappa:   Mean={km:.4f} → {k_str}")

    print(f"{'='*60}\n")

def print_seed_summary(seed, all_stage_metrics):
    print(f"\n{'#'*60}")
    print(f"Seed {seed} - All Stages Summary")
    print(f"{'#'*60}")

    header = f"  {'Size':<6} | {'Acc':<12} | {'Kappa':<12} | {'F1':<12} | {'AUC':<12}"
    print(header)
    print(f"  {'-'*58}")

    for size in sorted(all_stage_metrics.keys()):
        m = all_stage_metrics[size]
        print(f"  {size:<6} | "
              f"{np.mean(m['acc'])*100:.2f}±{np.std(m['acc'])*100:.2f}  | "
              f"{np.mean(m['kappa'])*100:.2f}±{np.std(m['kappa'])*100:.2f}  | "
              f"{np.mean(m['f1'])*100:.2f}±{np.std(m['f1'])*100:.2f}  | "
              f"{np.mean(m['auc'])*100:.2f}±{np.std(m['auc'])*100:.2f}")

    print(f"{'#'*60}\n")

def print_final_aggregation_report(all_results, seeds, data_sizes):
    print(f"\n{'*'*70}")
    print(f"Final Report (All Seeds × All Stages)")
    print(f"{'*'*70}")

    for size in data_sizes:
        print(f"\nData Size: {size}")
        print(f"{'-'*60}")

        aggregated = {}
        for seed in seeds:
            if seed in all_results and size in all_results[seed]:
                for metric_name, values in all_results[seed][size].items():
                    if metric_name not in aggregated:
                        aggregated[metric_name] = []
                    aggregated[metric_name].extend(values)

        if not aggregated:
            print(f"No data available.")
            continue

        for key in ["acc", "kappa", "mcc", "precision", "recall", "specificity",
                    "f1", "macro_f1", "auc", "auprc", "ece"]:
            if key in aggregated:
                vals = aggregated[key]
                print(f"    {key.upper():<9}: %{np.mean(vals)*100:.2f} ± {np.std(vals)*100:.2f}  "
                      f"(n={len(vals)}, {len(seeds)} seeds × {len(vals)//len(seeds)} folds)")

        kappa_all = aggregated.get("kappa", [])
        if kappa_all:
            km = np.mean(kappa_all)
            if km > 0.8: k_str = "Almost Perfect"
            elif km > 0.6: k_str = "Substantial"
            elif km > 0.4: k_str = "Moderate"
            elif km > 0.2: k_str = "Fair"
            else: k_str = "Slight/Poor"
            print(f"Cohen's Kappa: Mean={km:.4f} → {k_str}")

    print(f"\n{'*'*70}")
    print("All experiments completed!")
    print(f"{'*'*70}\n")

def save_results_json(results, filepath):
    serializable = {}
    for seed, sizes in results.items():
        serializable[str(seed)] = {}
        for size, metrics in sizes.items():
            serializable[str(seed)][str(size)] = {}
            for metric, values in metrics.items():
                serializable[str(seed)][str(size)][metric] = [float(v) for v in values]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved: {filepath}")
