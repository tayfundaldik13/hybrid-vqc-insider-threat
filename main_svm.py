"""
SVM Classifier Progressive Training Pipeline
"""
import os
import time
import random
import numpy as np
from sklearn.svm import SVC
from scipy import stats as sp_stats
import psutil, os as _os

from src import config
from src.data_process import (prepare_cert_embeddings,
    create_nonoverlapping_stages, create_nested_folds)
from src.utils import (
    calculate_metrics, plot_confusion_matrix,
    calculate_ece, plot_reliability_diagram,
    print_stage_report, print_seed_summary,
    print_final_aggregation_report, save_results_json
)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(BASE_DIR, "results_svm")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_svm")
CACHE_DIR      = os.path.join(BASE_DIR, "cache_svm")

SVM_C      = 0.113971
SVM_KERNEL = "linear"
SVM_GAMMA  = "auto"

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def train_and_evaluate(X_train, y_train, X_test, y_test, seed):
    model = SVC(
        C=SVM_C,
        kernel=SVM_KERNEL,
        gamma=SVM_GAMMA,
        probability=True,
        random_state=seed,
    )

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

    data_sizes = config.DATA_SIZES
    seeds      = config.SEEDS
    n_outer    = config.N_OUTER_FOLDS
    total_pool = sum(data_sizes)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"SVM Pipeline (Almusawy & Alrammahi 2024)")
    print(f"SVC={SVM_C} kernel={SVM_KERNEL} gamma={SVM_GAMMA}")
    print(f"Seeds: {seeds}  |  Data Sizes: {data_sizes}")
    print(f"{'='*60}\n")

    json_path = os.path.join(RESULTS_DIR, "svm_results.json")
    if os.path.exists(json_path):
        import json as _json
        with open(json_path) as _f:
            _existing = _json.load(_f)
        all_results = {int(k): {int(sz): v for sz, v in sv.items()}
                       for k, sv in _existing.items()}
        print(f"Loaded existing results for seeds: {sorted(all_results.keys())}")
    else:
        all_results = {}

    for seed_idx, seed in enumerate(seeds):
        if seed in all_results:
            print(f"\nSeed {seed} already done — skipping.")
            continue
        print(f"\n{'#'*60}")
        print(f"Seed {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'#'*60}")

        set_all_seeds(seed)
        full_embeddings, full_labels, full_texts = prepare_cert_embeddings(
            max_size=total_pool, seed=seed
        )
        stages = create_nonoverlapping_stages(
            full_embeddings, full_labels, data_sizes, seed=seed, texts=full_texts
        )
        for ds in data_sizes:
            s_emb, s_lab, _ = stages[ds]
            print(f"   Stage {ds}: {len(s_emb)} samples "
                  f"(threat={int(s_lab.sum())}, normal={int((s_lab==0).sum())})")

        all_results[seed] = {}

        for stage_idx, data_size in enumerate(data_sizes):
            stage_start = time.time()
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx+1}/{len(data_sizes)}: Size={data_size}, Seed={seed}")
            print(f"{'='*60}")

            stage_emb, stage_labels, _ = stages[data_size]
            nested_folds = create_nested_folds(stage_labels, n_outer, n_inner=1, seed=seed)

            stage_metrics = {
                "acc": [], "kappa": [], "mcc": [], "precision": [],
                "recall": [], "specificity": [], "f1": [], "macro_f1": [], "auc": [],
                "ece": [],
                "train_latency_s": [], "infer_latency_ms": [],
                "total_latency_s": [], "total_mem_mb": [], "n_params": []
            }
            fold_shapiro_ps = []

            for outer_idx, fold_data in enumerate(nested_folds):
                outer_train_idx = fold_data["outer_train"]
                outer_test_idx  = fold_data["outer_test"]

                print(f"\n  {'─'*50}")
                print(f"Outer Fold {outer_idx+1}/{n_outer}: "
                      f"Train={len(outer_train_idx)}, Test={len(outer_test_idx)}")

                X_train = stage_emb[outer_train_idx]
                X_test  = stage_emb[outer_test_idx]
                y_train = stage_labels[outer_train_idx]
                y_test  = stage_labels[outer_test_idx]

                set_all_seeds(seed)
                preds, probs, train_latency, infer_latency = train_and_evaluate(
                    X_train, y_train, X_test, y_test, seed
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
                fold_metrics["ece"]              = calculate_ece(y_test, probs)

                for key in stage_metrics:
                    if key in fold_metrics:
                        stage_metrics[key].append(fold_metrics[key])

                print(f"  >>> Outer {outer_idx+1} (train {train_latency:.3f}s): "
                      f"Acc={fold_metrics['acc']:.4f}  "
                      f"MCC={fold_metrics['mcc']:.4f}  "
                      f"F1={fold_metrics['f1']:.4f}  "
                      f"AUC={fold_metrics['auc']:.4f}")

                if outer_idx == n_outer - 1:
                    cm_path = os.path.join(
                        RESULTS_DIR, f"cm_svm_seed{seed}_size{data_size}.png")
                    plot_confusion_matrix(y_test, preds,
                        f"SVM CM (Size={data_size}, Seed={seed})",
                        save_path=cm_path)
                    rel_path = os.path.join(
                        RESULTS_DIR, f"reliability_svm_seed{seed}_size{data_size}.png")
                    plot_reliability_diagram(y_test, probs,
                        f"SVM Calibration (Size={data_size}, Seed={seed})",
                        save_path=rel_path)

            stage_time = time.time() - stage_start
            print(f"Stage {data_size} completed in {stage_time:.1f}s")

            f1_scores = stage_metrics.get("f1", [])
            if len(f1_scores) >= 3:
                _, p_sh = sp_stats.shapiro(f1_scores)
                fold_shapiro_ps.append(p_sh)

            print_stage_report(data_size, seed, stage_metrics,
                               fold_shapiro_ps=fold_shapiro_ps,
                               mann_whitney_p=1.0)
            all_results[seed][data_size] = stage_metrics

        save_results_json(all_results, json_path)
        print_seed_summary(seed, {ds: all_results[seed][ds] for ds in data_sizes})

    print_final_aggregation_report(all_results, seeds, data_sizes)
    save_results_json(all_results, json_path)

    total_time  = time.time() - pipeline_start
    peak_mem_mb = (peak_mem - mem_start) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"Total Pipeline Latency : {int(total_time//3600):02d}h "
          f"{int((total_time%3600)//60):02d}m {int(total_time%60):02d}s")
    print(f"{'='*60}")
    print(f"  Results: {json_path}")
    print("SVM Pipeline completed.")


if __name__ == "__main__":
    run_pipeline()
