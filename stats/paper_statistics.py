import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import shapiro, wilcoxon

MODELS = {
    "quantum": "Hybrid-VQC",
    "nb": "Naive Bayes",
    "svm": "SVM (Linear)",
    "singh": "BiLSTM",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "transformer": "Transformer",
    "reshybnet": "ResHybNet",
    "ssl": "SCARF",
}
ORDER = list(MODELS)
SIZES = ["50", "100", "250", "500"]
TEST_METRICS = ["f1", "recall", "auc"]
RANK_METRICS = ["f1", "recall", "auc", "mcc"]
ALPHA = 0.05


def find_results(root):
    found = {}
    for path in glob.glob(os.path.join(root, "**", "*_results.json"), recursive=True):
        if "_all" in os.path.basename(path) or "partial" in path:
            continue
        key = os.path.basename(path)[: -len("_results.json")]
        if key in MODELS and key not in found:
            found[key] = path
    missing = [k for k in ORDER if k not in found]
    if missing:
        sys.exit(f"{root}: missing results for {missing}")
    return found


def fold_vector(results, size, metric):
    values = []
    for seed in sorted(results, key=int):
        values += list(results[seed][size][metric])
    return np.asarray(values, dtype=float)


def cohens_d(a, b):
    ma, mb = round(a.mean() * 100, 2), round(b.mean() * 100, 2)
    sa, sb = round(a.std() * 100, 2), round(b.std() * 100, 2)
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
    return 0.0 if pooled == 0 else (ma - mb) / pooled


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, p[idx] * n / rank)
        q[idx] = running
    return q


def paired_p(a, b):
    if np.allclose(a, b):
        return 1.0
    return wilcoxon(a, b, alternative="two-sided").pvalue


def load(datasets):
    data = {}
    for name, root in datasets:
        paths = find_results(root)
        data[name] = {k: json.load(open(p)) for k, p in paths.items()}
        for k in ORDER:
            for s in SIZES:
                n = len(fold_vector(data[name][k], s, "f1"))
                if n != 50:
                    sys.exit(f"{name}/{k}/n={s}: expected 50 paired values, found {n}")
    return data


def wilcoxon_table(data):
    tests = []
    for ds in data:
        for s in SIZES:
            for m in TEST_METRICS:
                q = fold_vector(data[ds]["quantum"], s, m)
                for k in ORDER[1:]:
                    b = fold_vector(data[ds][k], s, m)
                    tests.append((ds, s, m, k, q.mean() - b.mean(), paired_p(q, b)))
    qvals = benjamini_hochberg([t[5] for t in tests])
    counts = {k: {m: [0, 0, 0] for m in TEST_METRICS} for k in ORDER[1:]}
    rows = []
    for (ds, s, m, k, diff, p), qv in zip(tests, qvals):
        if qv < ALPHA:
            idx = 0 if diff > 0 else 2
        else:
            idx = 1
        counts[k][m][idx] += 1
        rows.append((ds, s, m, MODELS[k], diff, p, qv, ["better", "n.s.", "worse"][idx]))
    return counts, rows, len(tests)


def rank_table(data):
    ranks = {k: {m: [] for m in RANK_METRICS} for k in ORDER}
    top3 = {k: 0 for k in ORDER}
    for ds in data:
        for s in SIZES:
            for m in RANK_METRICS:
                means = {k: fold_vector(data[ds][k], s, m).mean() for k in ORDER}
                ordered = sorted(ORDER, key=lambda k: -means[k])
                for r, k in enumerate(ordered, start=1):
                    ranks[k][m].append(r)
                if m == "f1":
                    for k in ordered[:3]:
                        top3[k] += 1
    return ranks, top3


def main():
    ap = argparse.ArgumentParser(description="Reproduce the statistical tables of the paper from the per-fold result JSONs.")
    ap.add_argument("--r42", required=True, help="CERT r4.2 results directory")
    ap.add_argument("--r52", required=True, help="CERT r5.2 results directory")
    ap.add_argument("--spedia", required=True, help="SPEDIA results directory")
    ap.add_argument("--out", default="statistics_output")
    args = ap.parse_args()

    datasets = [("CERT r4.2", args.r42), ("CERT r5.2", args.r52), ("SPEDIA", args.spedia)]
    data = load(datasets)
    os.makedirs(args.out, exist_ok=True)
    lines = []

    counts, rows, n_tests = wilcoxon_table(data)
    lines.append(f"TABLE XVIII - Wilcoxon signed-rank, Hybrid-VQC vs each baseline, BH over {n_tests} tests, q < {ALPHA}")
    lines.append(f"{'Baseline':16s} {'F1':>12s} {'Recall':>12s} {'AUC-ROC':>12s}")
    total = {m: [0, 0, 0] for m in TEST_METRICS}
    for k in ORDER[1:]:
        cells = []
        for m in TEST_METRICS:
            c = counts[k][m]
            total[m] = [x + y for x, y in zip(total[m], c)]
            cells.append(f"{c[0]} / {c[1]} / {c[2]}")
        lines.append(f"{MODELS[k]:16s} " + " ".join(f"{c:>12s}" for c in cells))
    lines.append(f"{'Total':16s} " + " ".join(f"{'{} / {} / {}'.format(*total[m]):>12s}" for m in TEST_METRICS))
    lines.append("")

    lines.append("TABLE XIX - Cohen's d for F1, Hybrid-VQC vs tree ensembles")
    lines.append(f"{'Dataset':10s} {'n':>4s} {'vs RF':>8s} {'vs XGB':>8s}")
    for ds in data:
        for s in SIZES:
            q = fold_vector(data[ds]["quantum"], s, "f1")
            d_rf = cohens_d(q, fold_vector(data[ds]["rf"], s, "f1"))
            d_xgb = cohens_d(q, fold_vector(data[ds]["xgboost"], s, "f1"))
            lines.append(f"{ds:10s} {s:>4s} {d_rf:>+8.2f} {d_xgb:>+8.2f}")
    lines.append("")

    ranks, top3 = rank_table(data)
    lines.append("TABLE XX - Mean rank over the 12 dataset-size settings and top-3 F1 finishes")
    lines.append(f"{'Model':16s} " + " ".join(f"{m.upper():>8s}" for m in RANK_METRICS) + f" {'Top-3 F1':>9s}")
    for k in sorted(ORDER, key=lambda k: np.mean(ranks[k]["f1"])):
        lines.append(f"{MODELS[k]:16s} " + " ".join(f"{np.mean(ranks[k][m]):>8.2f}" for m in RANK_METRICS) + f" {top3[k]:>9d}")
    lines.append("")

    r42 = data["CERT r4.2"]
    lines.append("TABLE XXI - Shapiro-Wilk on F1, CERT r4.2 (50 values per model and size)")
    for s in SIZES:
        for k in ORDER:
            w, p = shapiro(fold_vector(r42[k], s, "f1"))
            p_txt = "< 0.0001" if p < 1e-4 else f"{p:.4f}"
            lines.append(f"{s:>4s} {MODELS[k]:16s} W={w:.4f} p={p_txt:>9s} {'Non-normal' if p < ALPHA else 'Normal'}")
    lines.append("")

    lines.append("TABLE XXII - Cohen's d, Hybrid-VQC vs each baseline, CERT r4.2")
    for k in ORDER[1:]:
        for m in ["f1", "auc"]:
            ds = [cohens_d(fold_vector(r42["quantum"], s, m), fold_vector(r42[k], s, m)) for s in SIZES]
            lines.append(f"{MODELS[k]:16s} {m.upper():4s} " + " ".join(f"{d:>+7.2f}" for d in ds))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "paper_statistics.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "wilcoxon_all_tests.csv"), "w") as f:
        f.write("dataset,size,metric,baseline,mean_diff,p_value,q_value,outcome\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f},{r[5]:.6g},{r[6]:.6g},{r[7]}\n")


if __name__ == "__main__":
    main()
