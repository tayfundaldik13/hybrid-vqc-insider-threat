import os
import sys
import time
import random
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config as cfg
from src.data_process import prepare_spedia_embeddings, create_nonoverlapping_stages
from src.utils import (
    calculate_metrics, plot_confusion_matrix,
    calculate_ece, plot_reliability_diagram,
    print_stage_report, print_seed_summary,
    print_final_aggregation_report, save_results_json,
    _load_partial, _save_partial, _clear_partial,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    os.system(f"{sys.executable} -m pip install xgboost")
    from xgboost import XGBClassifier

RESULTS_DIR = cfg.RESULTS_XGBOOST_DIR

N_ESTIMATORS     = cfg.XGB_N_ESTIMATORS
MAX_DEPTH        = cfg.XGB_MAX_DEPTH
LEARNING_RATE    = cfg.XGB_LEARNING_RATE
SUBSAMPLE        = cfg.XGB_SUBSAMPLE
COLSAMPLE        = cfg.XGB_COLSAMPLE
MIN_CHILD_WEIGHT = cfg.XGB_MIN_CHILD_WEIGHT
SCALE_POS_WEIGHT = cfg.XGB_SCALE_POS_WEIGHT

DATA_SIZES = cfg.DATA_SIZES
SEEDS      = cfg.SEEDS
N_OUTER    = cfg.N_OUTER_FOLDS


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def train_and_evaluate(X_train, y_train, X_test, y_test, seed):
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        min_child_weight=MIN_CHILD_WEIGHT,
        scale_pos_weight=SCALE_POS_WEIGHT,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_latency = time.time() - t0

    t1 = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    infer_latency = time.time() - t1
    preds = (probs >= 0.5).astype(float)
    try:
        n_params = int(model.get_booster().trees_to_dataframe().shape[0])
    except Exception:
        n_params = N_ESTIMATORS * (2 ** MAX_DEPTH)
    return preds, probs, train_latency, infer_latency, n_params


def run_pipeline():
    import psutil, os as _os
    _proc = psutil.Process(_os.getpid())
    pipeline_start = time.time()
    mem_start = _proc.memory_info().rss
    peak_mem = mem_start

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("XGBoost PIPELINE - Staged 50-500 (SPEDIA)")
    print(f"   n_est={N_ESTIMATORS} depth={MAX_DEPTH} lr={LEARNING_RATE:.4f} spw={SCALE_POS_WEIGHT:.2f}")
    print(f"   subsample={SUBSAMPLE:.2f} colsample={COLSAMPLE:.2f} min_child_w={MIN_CHILD_WEIGHT}")
    print(f"   Seeds: {SEEDS}  |  Stages: {DATA_SIZES}  |  Nested CV: {N_OUTER} outer folds")
    print(f"{'='*60}\n")

    full_embeddings, full_labels = prepare_spedia_embeddings(cfg.SPEDIA_FEATURES_PATH)
    n_total = len(full_labels)
    print(f"   Full dataset: {n_total} samples  "
          f"(threat={int(full_labels.sum())}, "
          f"normal={int((full_labels==0).sum())}, "
          f"ratio={full_labels.mean():.4f})\n")

    json_path = os.path.join(RESULTS_DIR, "xgboost_results.json")
    if os.path.exists(json_path):
        import json as _json
        with open(json_path) as _f:
            _existing = _json.load(_f)
        all_results = {int(k): {int(sz): v for sz, v in sv.items()}
                       for k, sv in _existing.items()}
        print(f"Loaded existing results for seeds: {sorted(all_results.keys())}")
    else:
        all_results = {}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'#'*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
        print(f"{'#'*60}")

        if seed not in all_results:
            all_results[seed] = {}

        set_all_seeds(seed)
        stages = create_nonoverlapping_stages(
            full_embeddings, full_labels, DATA_SIZES, seed=seed, threat_ratio=0.4
        )
        for ds in DATA_SIZES:
            s_emb, s_lab = stages[ds]
            print(f"   Stage {ds}: {len(s_emb)} samples "
                  f"(threat={int(s_lab.sum())}, normal={int((s_lab==0).sum())})")

        for data_size in DATA_SIZES:
            if data_size in all_results[seed] and len(all_results[seed][data_size].get("f1", [])) == N_OUTER:
                print(f"\n   [SKIP] Size={data_size} already completed for seed {seed}.")
                continue

            print(f"\n{'='*60}")
            print(f"STAGE: Size={data_size}, Seed={seed}")
            print(f"{'='*60}")

            stage_emb, stage_labels = stages[data_size]
            from src.data_process import create_nested_folds
            nested_folds = create_nested_folds(stage_labels, N_OUTER, n_inner=1, seed=seed)

            stage_metrics = {
                "acc": [], "kappa": [], "mcc": [], "precision": [],
                "recall": [], "specificity": [], "f1": [], "macro_f1": [], "auc": [],
                "auprc": [], "ece": [],
                "train_latency_s": [], "infer_latency_ms": [],
                "total_latency_s": [], "total_mem_mb": [], "n_params": []
            }
            fold_shapiro_ps = []
            stage_start = time.time()

            _partial = _load_partial(RESULTS_DIR, seed, stage=data_size)
            _start_fold = 0
            if _partial is not None:
                _start_fold = _partial["next_fold"]
                _pm = _partial["stage_metrics"]
                for _k in stage_metrics:
                    if _k in _pm:
                        stage_metrics[_k] = list(_pm[_k])
                print(f"  [RESUME] Seed {seed} Size {data_size}: resuming from fold {_start_fold+1}")

            for outer_idx, fold_data in enumerate(nested_folds):
                if outer_idx < _start_fold:
                    continue

                outer_train_idx = fold_data["outer_train"]
                outer_test_idx  = fold_data["outer_test"]
                print(f"\n  {'-'*50}")
                print(f"  Outer Fold {outer_idx+1}/{N_OUTER}: "
                      f"Train={len(outer_train_idx)}, Test={len(outer_test_idx)}")

                mem_fold_start = _proc.memory_info().rss

                X_train = stage_emb[outer_train_idx]
                X_test  = stage_emb[outer_test_idx]
                y_train = stage_labels[outer_train_idx]
                y_test  = stage_labels[outer_test_idx]

                set_all_seeds(seed)
                preds, probs, train_latency, infer_latency, n_params = train_and_evaluate(
                    X_train, y_train, X_test, y_test, seed)

                current_rss = _proc.memory_info().rss
                if current_rss > peak_mem:
                    peak_mem = current_rss
                fold_mem_mb = max(0.0, (current_rss - mem_fold_start) / 1024 / 1024)

                fold_metrics = calculate_metrics(y_test, preds, probs)
                fold_metrics["train_latency_s"]  = train_latency
                fold_metrics["infer_latency_ms"] = infer_latency * 1000 / max(len(y_test), 1)
                fold_metrics["total_latency_s"]  = train_latency + infer_latency
                fold_metrics["total_mem_mb"]     = fold_mem_mb
                fold_metrics["n_params"]         = n_params
                fold_metrics["ece"]              = calculate_ece(y_test, probs)

                _pred_dir = os.path.join(RESULTS_DIR, "predictions")
                os.makedirs(_pred_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(_pred_dir, f"seed{seed}_size{data_size}_fold{outer_idx}.npz"),
                    y_true=y_test, y_pred=preds, test_idx=outer_test_idx)

                for key in stage_metrics:
                    if key in fold_metrics:
                        stage_metrics[key].append(fold_metrics[key])

                _save_partial(RESULTS_DIR, seed, outer_idx, stage_metrics, stage=data_size)

                print(f"  >>> Outer {outer_idx+1}: "
                      f"Acc={fold_metrics['acc']:.4f}  MCC={fold_metrics['mcc']:.4f}  "
                      f"F1={fold_metrics['f1']:.4f}  AUC={fold_metrics['auc']:.4f}")

                if outer_idx == N_OUTER - 1:
                    cm_path = os.path.join(RESULTS_DIR, f"cm_xgboost_seed{seed}_size{data_size}.png")
                    plot_confusion_matrix(y_test, preds,
                        f"XGBoost CM (Size={data_size}, Seed={seed})", save_path=cm_path)
                    rel_path = os.path.join(RESULTS_DIR, f"reliability_xgboost_seed{seed}_size{data_size}.png")
                    plot_reliability_diagram(y_test, probs,
                        f"XGBoost Reliability (Size={data_size}, Seed={seed})", save_path=rel_path)

            print(f"\n   Seed {seed} Size {data_size} done in {time.time()-stage_start:.1f}s")
            f1s = stage_metrics.get("f1", [])
            if len(f1s) >= 3:
                _, ps = sp_stats.shapiro(f1s)
                fold_shapiro_ps.append(ps)
            print_stage_report(data_size, seed, stage_metrics, fold_shapiro_ps=fold_shapiro_ps, mann_whitney_p=1.0)
            _clear_partial(RESULTS_DIR, seed, stage=data_size)
            all_results[seed][data_size] = stage_metrics
            save_results_json(all_results, json_path)

        print_seed_summary(seed, all_results[seed])

    print_final_aggregation_report(all_results, SEEDS, DATA_SIZES)
    save_results_json(all_results, json_path)
    total_time = time.time() - pipeline_start
    peak_mem_mb = (peak_mem - mem_start) / 1024 / 1024
    print(f"\n{'='*60}\n  Total time: {int(total_time//3600):02d}h{int((total_time%3600)//60):02d}m{int(total_time%60):02d}s"
          f"  Peak memory: {peak_mem_mb:.1f}MB\n  Results: {json_path}\n{'='*60}\nXGBoost (staged) done!")


if __name__ == "__main__":
    run_pipeline()
