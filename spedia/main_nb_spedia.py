"""
Naive Bayes — SPEDIA Dataset
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy import stats as sp_stats
import psutil, os as _os

import spedia_config as cfg
from data_process_spedia import (
    prepare_spedia_embeddings, create_nonoverlapping_stages, create_nested_folds
)
from src.utils import (
    calculate_metrics, plot_confusion_matrix,
    calculate_ece, plot_reliability_diagram,
    print_stage_report, print_seed_summary,
    print_final_aggregation_report, save_results_json
)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = GaussianNB(var_smoothing=cfg.VAR_SMOOTHING)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_latency = time.time() - t0
    t1 = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    infer_latency = time.time() - t1
    preds = (probs >= 0.5).astype(float)
    return preds, probs, train_latency, infer_latency


def run_pipeline():
    _proc = psutil.Process(_os.getpid())
    pipeline_start = time.time()
    mem_start = _proc.memory_info().rss
    peak_mem  = mem_start

    os.makedirs(cfg.RESULTS_NB_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_SINGH_DIR, exist_ok=True)

    data_sizes = cfg.DATA_SIZES
    seeds      = cfg.SEEDS
    n_outer    = cfg.N_OUTER_FOLDS

    print(f"SPEDIA - Naive Bayes")
    print(f"var_smoothing={cfg.VAR_SMOOTHING:.2e}")
    print(f"Seeds: {seeds}  |  Data Sizes: {data_sizes}")
    print(f"{'='*60}\n")

    json_path = os.path.join(cfg.RESULTS_NB_DIR, "nb_results.json")
    if os.path.exists(json_path):
        import json as _json
        with open(json_path) as _f:
            _existing = _json.load(_f)
        all_results = {int(k): {int(sz): v for sz, v in sv.items()}
                       for k, sv in _existing.items()}
        print(f"Loaded existing results for seeds: {sorted(all_results.keys())}")
    else:
        all_results = {}

    full_embeddings, full_labels, full_texts = prepare_spedia_embeddings(cfg.SPEDIA_FEATURES_PATH)

    for seed_idx, seed in enumerate(seeds):
        if seed in all_results:
            print(f"\nSeed {seed} already done — skipping.")
            continue
        print(f"\n{'#'*60}")
        print(f"Seed {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'#'*60}")

        set_all_seeds(seed)
        stages = create_nonoverlapping_stages(
            full_embeddings, full_labels, data_sizes,
            seed=seed, texts=full_texts, threat_ratio=cfg.THREAT_RATIO
        )
        for ds in data_sizes:
            s_emb, s_lab, _ = stages[ds]
            print(f"Stage {ds}: {len(s_emb)} samples "
                  f"(threat={int(s_lab.sum())}, normal={int((s_lab==0).sum())})")

        all_results[seed] = {}

        for stage_idx, data_size in enumerate(data_sizes):
            stage_start = time.time()
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx+1}/{len(data_sizes)}: Size={data_size}, Seed={seed}")
            print(f"{'='*60}")

            stage_emb, stage_labels, _ = stages[data_size]
            nested_folds = create_nested_folds(stage_labels, n_outer, seed=seed)

            stage_metrics = {
                "acc": [], "kappa": [], "mcc": [], "precision": [],
                "recall": [], "specificity": [], "f1": [], "macro_f1": [],
                "auc": [], "auprc": [], "ece": [],
                "train_latency_s": [], "infer_latency_ms": [],
                "total_latency_s": [], "total_mem_mb": [], "n_params": []
            }
            fold_shapiro_ps = []

            for outer_idx, fold_data in enumerate(nested_folds):
                outer_train_idx = fold_data["outer_train"]
                outer_test_idx  = fold_data["outer_test"]

                print(f"\n  Outer Fold {outer_idx+1}/{n_outer}: "
                      f"Train={len(outer_train_idx)}, Test={len(outer_test_idx)}")

                X_train = stage_emb[outer_train_idx]
                X_test  = stage_emb[outer_test_idx]
                y_train = stage_labels[outer_train_idx]
                y_test  = stage_labels[outer_test_idx]

                set_all_seeds(seed)
                preds, probs, train_latency, infer_latency = train_and_evaluate(
                    X_train, y_train, X_test, y_test
                )

                current_rss = _proc.memory_info().rss
                if current_rss > peak_mem:
                    peak_mem = current_rss

                fold_metrics = calculate_metrics(y_test, preds, probs)
                fold_metrics["train_latency_s"]  = train_latency
                fold_metrics["infer_latency_ms"] = infer_latency * 1000 / max(len(y_test), 1)
                fold_metrics["total_latency_s"]  = train_latency + infer_latency
                fold_metrics["total_mem_mb"]     = 0.0
                fold_metrics["n_params"]         = 0

                for key in stage_metrics:
                    if key in fold_metrics:
                        stage_metrics[key].append(fold_metrics[key])

                print(f"  >>> Fold {outer_idx+1}: "
                      f"Acc={fold_metrics['acc']:.4f}  "
                      f"MCC={fold_metrics['mcc']:.4f}  "
                      f"F1={fold_metrics['f1']:.4f}  "
                      f"AUC={fold_metrics['auc']:.4f}")

                if outer_idx == n_outer - 1:
                    plot_confusion_matrix(y_test, preds,
                        f"NB CM (Size={data_size}, Seed={seed})",
                        save_path=os.path.join(cfg.RESULTS_NB_DIR,
                            f"cm_nb_seed{seed}_size{data_size}.png"))
                    plot_reliability_diagram(y_test, probs,
                        f"NB Calibration (Size={data_size}, Seed={seed})",
                        save_path=os.path.join(cfg.RESULTS_NB_DIR,
                            f"reliability_nb_seed{seed}_size{data_size}.png"))

            stage_time = time.time() - stage_start
            print(f" Stage {data_size} done in {stage_time:.1f}s")

            f1_scores = stage_metrics.get("f1", [])
            if len(f1_scores) >= 3:
                _, p_sh = sp_stats.shapiro(f1_scores)
                fold_shapiro_ps.append(p_sh)

            print_stage_report(data_size, seed, stage_metrics,
                               fold_shapiro_ps=fold_shapiro_ps, mann_whitney_p=1.0)
            all_results[seed][data_size] = stage_metrics

        save_results_json(all_results, json_path)
        print_seed_summary(seed, {ds: all_results[seed][ds] for ds in data_sizes})

    print_final_aggregation_report(all_results, seeds, data_sizes)
    save_results_json(all_results, json_path)

    total_time  = time.time() - pipeline_start
    peak_mem_mb = (peak_mem - mem_start) / 1024 / 1024
    print(f"\nTotal: {int(total_time//60)}m {int(total_time%60)}s | "
          f"Peak mem: {peak_mem_mb:.1f} MB")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_pipeline()
