import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import json
import shutil
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats as sp_stats

torch.set_num_threads(1)

import main_quantum as mq
from main_quantum import config, qconfig
from paper_arch import check_config as _check_arch
_check_arch(qconfig, "r4.2")

SMOKE = "--smoke" in sys.argv
_ARGS = [a for a in sys.argv[1:] if a != "--smoke"]

RESULTS_DIR = qconfig.QUANTUM_RESULTS_DIR + ("_smoke" if SMOKE else "")
CKPT_DIR    = qconfig.QUANTUM_CHECKPOINT_DIR + ("_smoke" if SMOKE else "")
FOLD_DIR    = os.path.join(RESULTS_DIR, "folds")
PRED_DIR    = os.path.join(RESULTS_DIR, "predictions")

METRIC_KEYS = ["acc", "kappa", "mcc", "precision", "recall", "specificity", "f1", "macro_f1", "auc",
               "ece", "train_latency_s", "infer_latency_ms", "total_latency_s", "total_mem_mb", "n_params"]


def _fold_json(seed, size, fold):
    return os.path.join(FOLD_DIR, f"seed{seed}_size{size}_fold{fold}.json")


def worker(seed: int, stage_idx: int, outer_idx: int):
    import psutil
    from src.data_process import prepare_cert_embeddings, create_nonoverlapping_stages, create_nested_folds
    from src.utils import calculate_ece

    data_sizes = config.DATA_SIZES
    data_size = data_sizes[stage_idx]
    n_outer = config.N_OUTER_FOLDS

    if os.path.exists(_fold_json(seed, data_size, outer_idx)):
        print(f"[seed {seed} size {data_size} fold {outer_idx}] already done - skipping")
        return
    for d in (FOLD_DIR, PRED_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)

    proc = psutil.Process(os.getpid())

    mq.set_all_seeds(seed)
    full_embeddings, full_labels = prepare_cert_embeddings(max_size=sum(data_sizes), seed=seed)
    stages = create_nonoverlapping_stages(full_embeddings, full_labels, data_sizes, seed=seed)
    stage_emb, stage_labels = stages[data_size]

    nested_folds = create_nested_folds(stage_labels, n_outer, n_inner=1, seed=seed)
    fold_data = nested_folds[outer_idx]
    outer_train_idx = fold_data["outer_train"]
    outer_test_idx = fold_data["outer_test"]

    n_trials = qconfig.OPTUNA_INNER_TRIALS
    if SMOKE:
        rng_s = np.random.default_rng(0)
        outer_train_idx = rng_s.choice(outer_train_idx, min(600, len(outer_train_idx)), replace=False)
        outer_test_idx = rng_s.choice(outer_test_idx, min(200, len(outer_test_idx)), replace=False)
        n_trials = 1

    print(f"[seed {seed} size {data_size} fold {outer_idx+1}/{n_outer}] "
          f"Train={len(outer_train_idx)} (threat={int(stage_labels[outer_train_idx].sum())}), "
          f"Test={len(outer_test_idx)} (threat={int(stage_labels[outer_test_idx].sum())})", flush=True)
    mem_fold_start = proc.memory_info().rss

    opt_start = time.time()
    best_params, best_threshold = mq.run_inner_optuna(
        stage_emb[outer_train_idx], stage_labels[outer_train_idx], seed=seed, n_trials=n_trials)
    opt_time = time.time() - opt_start
    print(f"   Optuna ({n_trials} trials, {opt_time:.0f}s): {best_params} threshold={best_threshold:.2f}", flush=True)

    X_out_train = torch.tensor(stage_emb[outer_train_idx],    dtype=torch.float32)
    X_out_test  = torch.tensor(stage_emb[outer_test_idx],     dtype=torch.float32)
    y_out_train = torch.tensor(stage_labels[outer_train_idx], dtype=torch.float32)
    y_out_test  = torch.tensor(stage_labels[outer_test_idx],  dtype=torch.float32)

    eff_bs = max(2, min(best_params["batch_size"], len(X_out_train) - 1))
    outer_dl = DataLoader(TensorDataset(X_out_train, y_out_train),
                          batch_size=eff_bs, shuffle=True, drop_last=True)

    mq.set_all_seeds(seed)
    model = mq.HybridQuantumModel(
        input_dim=qconfig.INPUT_DIM, n_qubits=qconfig.N_QUBITS, n_layers=qconfig.N_LAYERS,
        nn_hidden=qconfig.NN_HIDDEN, nn_depth=qconfig.NN_DEPTH, pre_hidden=qconfig.PRE_HIDDEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
    loss_fn   = nn.BCEWithLogitsLoss()

    t_start = time.time()
    outer_metrics, targets, preds, outer_probs = mq.train_one_fold(
        model, optimizer, loss_fn, outer_dl, X_out_test, y_out_test,
        best_params["epochs"], threshold=best_threshold)
    fold_time = time.time() - t_start

    outer_metrics["total_mem_mb"] = max(0.0, (proc.memory_info().rss - mem_fold_start) / 1024 / 1024)
    outer_metrics["ece"] = calculate_ece(targets, outer_probs)

    mq.save_hybrid_checkpoint(model, optimizer, best_params["epochs"], f"fold{outer_idx}",
                              data_size, seed, CKPT_DIR)
    np.savez_compressed(os.path.join(PRED_DIR, f"seed{seed}_size{data_size}_fold{outer_idx}.npz"),
                        y_true=targets, y_pred=preds, y_prob=outer_probs, test_idx=outer_test_idx)

    out = {"metrics": {k: float(v) for k, v in outer_metrics.items()},
           "threshold": float(best_threshold), "best_params": best_params,
           "opt_time_s": opt_time, "fold_time_s": fold_time}
    tmp = _fold_json(seed, data_size, outer_idx) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, _fold_json(seed, data_size, outer_idx))

    print(f">>> seed {seed} size {data_size} fold {outer_idx+1} done (train {fold_time:.0f}s): "
          f"Acc={outer_metrics['acc']:.4f} MCC={outer_metrics['mcc']:.4f} F1={outer_metrics['f1']:.4f} "
          f"MacroF1={outer_metrics['macro_f1']:.4f} AUC={outer_metrics['auc']:.4f} "
          f"ECE={outer_metrics['ece']:.4f}", flush=True)


def aggregate(seeds):
    from src.utils import (print_stage_report, print_seed_summary, print_final_aggregation_report,
                           save_results_json, plot_confusion_matrix, plot_reliability_diagram)

    data_sizes = config.DATA_SIZES
    n_outer = config.N_OUTER_FOLDS
    json_path = os.path.join(RESULTS_DIR, "quantum_results.json")
    all_results = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_results = {int(k): {int(sz): v for sz, v in sv.items()} for k, sv in json.load(f).items()}

    for seed in seeds:
        for data_size in data_sizes:
            if data_size in all_results.get(seed, {}):
                continue
            missing = [i for i in range(n_outer) if not os.path.exists(_fold_json(seed, data_size, i))]
            if missing:
                print(f"seed {seed} size {data_size}: folds {missing} not finished - skipping aggregation")
                continue
            folds = []
            for i in range(n_outer):
                with open(_fold_json(seed, data_size, i)) as f:
                    folds.append(json.load(f))

            stage_metrics = {k: [] for k in METRIC_KEYS}
            for fd in folds:
                for k in stage_metrics:
                    if k in fd["metrics"]:
                        stage_metrics[k].append(fd["metrics"][k])

            f1s = stage_metrics["f1"]
            shapiro = [sp_stats.shapiro(f1s)[1]] if len(f1s) >= 3 else []
            print(f"\nSeed {seed} size {data_size}: mean F1-max threshold = "
                  f"{np.mean([fd['threshold'] for fd in folds]):.3f}")
            print_stage_report(data_size, seed, stage_metrics, fold_shapiro_ps=shapiro, mann_whitney_p=1.0)

            last = np.load(os.path.join(PRED_DIR, f"seed{seed}_size{data_size}_fold{n_outer-1}.npz"))
            plot_confusion_matrix(last["y_true"], last["y_pred"], f"Quantum CM (Size={data_size}, Seed={seed})",
                save_path=os.path.join(RESULTS_DIR, f"cm_quantum_seed{seed}_size{data_size}.png"))
            plot_reliability_diagram(last["y_true"], last["y_prob"],
                f"Quantum Calibration (Size={data_size}, Seed={seed})",
                save_path=os.path.join(RESULTS_DIR, f"reliability_quantum_seed{seed}_size{data_size}.png"))

            best_i = int(np.argmax(f1s))
            src_ck = mq.get_hybrid_checkpoint_path(CKPT_DIR, seed, data_size, f"fold{best_i}")
            dst_ck = mq.get_hybrid_checkpoint_path(CKPT_DIR, seed, data_size, "best")
            if os.path.exists(src_ck):
                shutil.copyfile(src_ck, dst_ck)

            all_results.setdefault(seed, {})[data_size] = stage_metrics
            save_results_json(all_results, json_path)

    done = [s for s in seeds if all(ds in all_results.get(s, {}) for ds in data_sizes)]
    for s in done:
        print_seed_summary(s, {ds: all_results[s][ds] for ds in data_sizes})
    if done:
        print_final_aggregation_report(all_results, done, data_sizes)


if __name__ == "__main__":
    mode = _ARGS[0]
    if mode == "worker":
        worker(int(_ARGS[1]), int(_ARGS[2]), int(_ARGS[3]))
    elif mode == "aggregate":
        aggregate([int(s) for s in _ARGS[1:]])
    else:
        sys.exit(f"unknown mode {mode}")
