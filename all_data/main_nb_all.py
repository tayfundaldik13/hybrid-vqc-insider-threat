import os
import sys
import time
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy import stats as sp_stats

_ALL_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_ALL_DATA_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _ALL_DATA_DIR)

import importlib.util as _ilu
_cfg = _ilu.spec_from_file_location("src.config", os.path.join(_ALL_DATA_DIR, "src", "config.py"))
config = _ilu.module_from_spec(_cfg); _cfg.loader.exec_module(config)
del _ilu, _cfg
from src.data_process import load_behavioral_features, create_nested_folds
from src.utils import (
    calculate_metrics, plot_confusion_matrix,
    calculate_ece, plot_reliability_diagram,
    print_stage_report, print_final_aggregation_report, save_results_json,
    _load_partial, _save_partial, _clear_partial
)

RESULTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_all", "nb")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_all", "nb")

VAR_SMOOTHING = config.NB_VAR_SMOOTHING
STAGE_KEY     = "all"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = GaussianNB(var_smoothing=VAR_SMOOTHING)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_latency = time.time() - t0
    t1 = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    infer_latency = time.time() - t1
    preds = (probs >= 0.5).astype(float)
    return preds, probs, train_latency, infer_latency


def run_pipeline():
    import psutil, os as _os
    _proc = psutil.Process(_os.getpid())
    pipeline_start = time.time()
    mem_start = _proc.memory_info().rss
    peak_mem  = mem_start

    seeds   = config.SEEDS
    n_outer = config.N_OUTER_FOLDS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("NAIVE BAYES PIPELINE - FULL DATASET (Dosh 2021)")
    print(f"   GaussianNB var_smoothing={VAR_SMOOTHING:.2e}")
    print(f"   Seeds: {seeds}  |  Nested CV: {n_outer} outer folds")
    print(f"{'='*60}\n")

    full_embeddings, full_labels = load_behavioral_features()
    n_total = len(full_labels)
    print(f"   Full dataset: {n_total} samples  "
          f"(threat={int(full_labels.sum())}, "
          f"normal={int((full_labels==0).sum())}, "
          f"ratio={full_labels.mean():.4f})\n")

    json_path = os.path.join(RESULTS_DIR, "nb_all_results.json")
    if os.path.exists(json_path):
        import json as _json
        with open(json_path) as _f:
            _existing = _json.load(_f)
        all_results = {int(k): v for k, v in _existing.items()}
        print(f"Loaded existing results for seeds: {sorted(all_results.keys())}")
    else:
        all_results = {}

    for seed_idx, seed in enumerate(seeds):
        if seed in all_results:
            print(f"\nSEED {seed} already done - skipping.")
            continue
        print(f"\n{'#'*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'#'*60}")

        set_all_seeds(seed)
        nested_folds = create_nested_folds(full_labels, n_outer, n_inner=1, seed=seed)

        stage_metrics = {
            "acc": [], "kappa": [], "mcc": [], "precision": [],
            "recall": [], "specificity": [], "f1": [], "macro_f1": [], "auc": [],
            "ece": [], "train_latency_s": [], "infer_latency_ms": [],
            "total_latency_s": [], "total_mem_mb": [], "n_params": []
        }
        fold_shapiro_ps = []
        stage_start     = time.time()

        _partial = _load_partial(RESULTS_DIR, seed)
        _start_fold = 0
        if _partial is not None:
            _start_fold = _partial["next_fold"]
            _pm = _partial["stage_metrics"]
            for _k in stage_metrics:
                if _k in _pm:
                    stage_metrics[_k] = list(_pm[_k])
            print(f"  [RESUME] Seed {seed}: resuming from fold {_start_fold+1}")

        for outer_idx, fold_data in enumerate(nested_folds):
            outer_train_idx = fold_data["outer_train"]
            if outer_idx < _start_fold:
                continue
            outer_test_idx  = fold_data["outer_test"]
            print(f"\n  {'-'*50}")
            print(f"  Outer Fold {outer_idx+1}/{n_outer}: "
                  f"Train={len(outer_train_idx)}, Test={len(outer_test_idx)}")

            mem_fold_start = _proc.memory_info().rss

            X_train = full_embeddings[outer_train_idx]
            X_test  = full_embeddings[outer_test_idx]
            y_train = full_labels[outer_train_idx]
            y_test  = full_labels[outer_test_idx]

            set_all_seeds(seed)
            preds, probs, train_latency, infer_latency = train_and_evaluate(
                X_train, y_train, X_test, y_test)

            current_rss = _proc.memory_info().rss
            if current_rss > peak_mem:
                peak_mem = current_rss
            fold_mem_mb = max(0.0, (current_rss - mem_fold_start) / 1024 / 1024)

            fold_metrics = calculate_metrics(y_test, preds, probs)
            fold_metrics["train_latency_s"]  = train_latency
            fold_metrics["infer_latency_ms"] = infer_latency * 1000 / max(len(y_test), 1)
            fold_metrics["total_latency_s"]  = train_latency + infer_latency
            fold_metrics["total_mem_mb"]     = fold_mem_mb
            fold_metrics["n_params"]         = 0
            fold_metrics["ece"]              = calculate_ece(y_test, probs)

            _pred_dir = os.path.join(RESULTS_DIR, "predictions")
            os.makedirs(_pred_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(_pred_dir, f"seed{seed}_fold{outer_idx}.npz"),
                y_true=y_test, y_pred=preds, test_idx=outer_test_idx)

            for key in stage_metrics:
                if key in fold_metrics:
                    stage_metrics[key].append(fold_metrics[key])

            _save_partial(RESULTS_DIR, seed, outer_idx, stage_metrics)

            print(f"  >>> Outer {outer_idx+1}: "
                  f"Acc={fold_metrics['acc']:.4f}  MCC={fold_metrics['mcc']:.4f}  "
                  f"F1={fold_metrics['f1']:.4f}  AUC={fold_metrics['auc']:.4f}")

            if outer_idx == n_outer - 1:
                cm_path = os.path.join(RESULTS_DIR, f"cm_nb_all_seed{seed}.png")
                plot_confusion_matrix(y_test, preds,
                    f"NB CM (Full Data, Seed={seed})", save_path=cm_path)
                rel_path = os.path.join(RESULTS_DIR, f"reliability_nb_all_seed{seed}.png")
                plot_reliability_diagram(y_test, probs,
                    f"NB Calibration (Full Data, Seed={seed})", save_path=rel_path)

        print(f"\n   Seed {seed} completed in {time.time()-stage_start:.1f}s")

        f1_scores = stage_metrics.get("f1", [])
        if len(f1_scores) >= 3:
            _, p_sh = sp_stats.shapiro(f1_scores)
            fold_shapiro_ps.append(p_sh)

        print_stage_report(STAGE_KEY, seed, stage_metrics,
                           fold_shapiro_ps=fold_shapiro_ps, mann_whitney_p=1.0)

        _clear_partial(RESULTS_DIR, seed)
        all_results[seed] = {STAGE_KEY: stage_metrics}
        save_results_json(all_results, json_path)

    print_final_aggregation_report(all_results, seeds, [STAGE_KEY])
    save_results_json(all_results, json_path)

    total_time  = time.time() - pipeline_start
    peak_mem_mb = (peak_mem - mem_start) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"  Total Latency : {int(total_time//3600):02d}h "
          f"{int((total_time%3600)//60):02d}m {int(total_time%60):02d}s")
    print(f"  Peak Memory   : {peak_mem_mb:.1f} MB")
    print(f"  Results       : {json_path}")
    print(f"{'='*60}")
    print("Naive Bayes (Full Data) Pipeline completed!")


if __name__ == "__main__":
    run_pipeline()
