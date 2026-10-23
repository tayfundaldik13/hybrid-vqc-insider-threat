import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

import main_quantum_all as mq
from main_quantum_all import config, qconfig
from paper_arch import check_config as _check_arch
_check_arch(qconfig, "r4.2")

SMOKE = "--smoke" in sys.argv
_ARGS = [a for a in sys.argv[1:] if a != "--smoke"]

RESULTS_DIR = mq.RESULTS_DIR + ("_smoke" if SMOKE else "")
CKPT_DIR    = mq.CHECKPOINT_DIR + ("_smoke" if SMOKE else "")
FOLD_DIR    = os.path.join(RESULTS_DIR, "folds")
PRED_DIR    = os.path.join(RESULTS_DIR, "predictions")
STAGE_KEY   = mq.STAGE_KEY


def _fold_json(seed, fold):
    return os.path.join(FOLD_DIR, f"seed{seed}_fold{fold}.json")


def worker(seed: int, outer_idx: int):
    import psutil
    from src.data_process import load_behavioral_features, create_nested_folds
    from src.utils import calculate_ece

    if os.path.exists(_fold_json(seed, outer_idx)):
        print(f"[seed {seed} fold {outer_idx}] already done - skipping")
        return
    os.makedirs(FOLD_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    proc = psutil.Process(os.getpid())
    n_outer = config.N_OUTER_FOLDS

    full_embeddings, full_labels = load_behavioral_features()
    mq.set_all_seeds(seed)
    nested_folds = create_nested_folds(full_labels, n_outer, n_inner=1, seed=seed)
    fold_data = nested_folds[outer_idx]
    outer_train_idx = fold_data["outer_train"]
    outer_test_idx = fold_data["outer_test"]

    n_trials = qconfig.OPTUNA_INNER_TRIALS
    if SMOKE:
        rng_s = np.random.default_rng(0)
        outer_train_idx = rng_s.choice(outer_train_idx, 600, replace=False)
        outer_test_idx = rng_s.choice(outer_test_idx, 200, replace=False)
        n_trials = 1

    print(f"[seed {seed} fold {outer_idx+1}/{n_outer}] "
          f"Train={len(outer_train_idx)}, Test={len(outer_test_idx)}", flush=True)
    mem_fold_start = proc.memory_info().rss

    opt_start = time.time()
    best_params, best_threshold = mq.run_inner_optuna(
        full_embeddings[outer_train_idx], full_labels[outer_train_idx],
        seed=seed, n_trials=n_trials)
    opt_time = time.time() - opt_start
    print(f"   Optuna ({n_trials} trials, {opt_time:.0f}s): {best_params} "
          f"threshold={best_threshold:.2f}", flush=True)

    X_out_train = torch.tensor(full_embeddings[outer_train_idx], dtype=torch.float32)
    X_out_test  = torch.tensor(full_embeddings[outer_test_idx],  dtype=torch.float32)
    y_out_train = torch.tensor(full_labels[outer_train_idx],     dtype=torch.float32)
    y_out_test  = torch.tensor(full_labels[outer_test_idx],      dtype=torch.float32)

    eff_bs = max(2, min(best_params["batch_size"], len(X_out_train) - 1))

    y_otr_np = full_labels[outer_train_idx]
    pos_i = np.where(y_otr_np == 1)[0]; neg_i = np.where(y_otr_np == 0)[0]
    n_qval = max(2, int(len(y_otr_np) * config.EARLY_STOP_VAL_FRAC))
    n_vp   = max(1, int(n_qval * y_otr_np.mean())); n_vn = n_qval - n_vp
    rng_q  = np.random.default_rng(seed)
    qval_idx = np.concatenate([rng_q.choice(pos_i, min(n_vp, len(pos_i)), replace=False),
                               rng_q.choice(neg_i, min(n_vn, len(neg_i)), replace=False)])
    qtr_idx  = np.setdiff1d(np.arange(len(y_otr_np)), qval_idx)
    X_qval = X_out_train[qval_idx]; y_qval = y_out_train[qval_idx]
    X_qtr  = X_out_train[qtr_idx];  y_qtr  = y_out_train[qtr_idx]
    val_pw_q = torch.tensor([float((y_qval.numpy() == 0).sum() /
                                   max((y_qval.numpy() == 1).sum(), 1))],
                            dtype=torch.float32)
    q_val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=val_pw_q)
    eff_bs2 = max(2, min(eff_bs, len(X_qtr) - 1))
    outer_dl = DataLoader(TensorDataset(X_qtr, y_qtr),
                          batch_size=eff_bs2, shuffle=True, drop_last=True)

    mq.set_all_seeds(seed)
    model = mq.HybridQuantumModel(
        input_dim=qconfig.INPUT_DIM, n_qubits=qconfig.N_QUBITS,
        n_layers=qconfig.N_LAYERS, nn_hidden=qconfig.NN_HIDDEN,
        nn_depth=qconfig.NN_DEPTH, pre_hidden=qconfig.PRE_HIDDEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
    loss_fn   = nn.BCEWithLogitsLoss()

    t_start = time.time()
    outer_metrics, targets, preds, outer_probs = mq.train_one_fold(
        model, optimizer, loss_fn, outer_dl, X_out_test, y_out_test,
        best_params["epochs"], threshold=best_threshold,
        X_val=X_qval, y_val=y_qval, val_loss_fn=q_val_loss_fn,
        patience=config.EARLY_STOP_PATIENCE, min_delta=config.EARLY_STOP_MIN_DELTA)
    fold_time = time.time() - t_start

    outer_metrics["total_mem_mb"] = max(
        0.0, (proc.memory_info().rss - mem_fold_start) / 1024 / 1024)
    outer_metrics["ece"] = calculate_ece(targets, outer_probs)

    mq.save_hybrid_checkpoint(model, optimizer, best_params["epochs"],
                              f"fold{outer_idx}", STAGE_KEY, seed, CKPT_DIR)
    np.savez_compressed(os.path.join(PRED_DIR, f"seed{seed}_fold{outer_idx}.npz"),
                        y_true=targets, y_pred=preds, y_prob=outer_probs,
                        test_idx=outer_test_idx)

    out = {"metrics": {k: float(v) for k, v in outer_metrics.items()},
           "threshold": float(best_threshold),
           "best_params": best_params,
           "opt_time_s": opt_time, "fold_time_s": fold_time}
    tmp = _fold_json(seed, outer_idx) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, _fold_json(seed, outer_idx))

    print(f">>> seed {seed} fold {outer_idx+1} done (train {fold_time:.0f}s): "
          f"Acc={outer_metrics['acc']:.4f} MCC={outer_metrics['mcc']:.4f} "
          f"F1={outer_metrics['f1']:.4f} AUC={outer_metrics['auc']:.4f} "
          f"ECE={outer_metrics['ece']:.4f}", flush=True)


def aggregate(seeds):
    from src.utils import (print_stage_report, save_results_json,
                           plot_confusion_matrix, plot_reliability_diagram)

    n_outer = config.N_OUTER_FOLDS
    json_path = os.path.join(RESULTS_DIR, "quantum_all_results.json")
    all_results = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_results = {int(k): v for k, v in json.load(f).items()}

    for seed in seeds:
        missing = [i for i in range(n_outer) if not os.path.exists(_fold_json(seed, i))]
        if missing:
            print(f"seed {seed}: folds {missing} not finished - skipping aggregation")
            continue
        folds = []
        for i in range(n_outer):
            with open(_fold_json(seed, i)) as f:
                folds.append(json.load(f))

        stage_metrics = {k: [] for k in [
            "acc", "kappa", "mcc", "precision", "recall", "specificity", "f1",
            "macro_f1", "auc", "ece", "train_latency_s", "infer_latency_ms",
            "total_latency_s", "total_mem_mb", "n_params"]}
        for fd in folds:
            for k in stage_metrics:
                if k in fd["metrics"]:
                    stage_metrics[k].append(fd["metrics"][k])

        f1s = stage_metrics["f1"]
        shapiro = [sp_stats.shapiro(f1s)[1]] if len(f1s) >= 3 else []
        print(f"\nSeed {seed}: mean F1-max threshold = "
              f"{np.mean([fd['threshold'] for fd in folds]):.3f}")
        print_stage_report(STAGE_KEY, seed, stage_metrics,
                           fold_shapiro_ps=shapiro, mann_whitney_p=1.0)

        last = np.load(os.path.join(PRED_DIR, f"seed{seed}_fold{n_outer-1}.npz"))
        plot_confusion_matrix(last["y_true"], last["y_pred"],
            f"Quantum CM (Full Data, Seed={seed})",
            save_path=os.path.join(RESULTS_DIR, f"cm_quantum_all_seed{seed}.png"))
        plot_reliability_diagram(last["y_true"], last["y_prob"],
            f"Quantum Calibration (Full Data, Seed={seed})",
            save_path=os.path.join(RESULTS_DIR, f"reliability_quantum_all_seed{seed}.png"))

        best_i = int(np.argmax(f1s))
        src_ck = mq.get_hybrid_checkpoint_path(CKPT_DIR, seed, STAGE_KEY, f"fold{best_i}")
        dst_ck = mq.get_hybrid_checkpoint_path(CKPT_DIR, seed, STAGE_KEY, "best")
        if os.path.exists(src_ck):
            shutil.copyfile(src_ck, dst_ck)

        all_results[seed] = {STAGE_KEY: stage_metrics}
        save_results_json(all_results, json_path)


if __name__ == "__main__":
    mode = _ARGS[0]
    if mode == "worker":
        worker(int(_ARGS[1]), int(_ARGS[2]))
    elif mode == "aggregate":
        aggregate([int(s) for s in _ARGS[1:]])
    else:
        sys.exit(f"unknown mode {mode}")
