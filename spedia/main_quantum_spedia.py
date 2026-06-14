"""
Hybrid Quantum VQC — SPEDIA Dataset
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from scipy import stats as sp_stats

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

import spedia_config as cfg
from data_process_spedia import (
    prepare_spedia_embeddings, create_nonoverlapping_stages, create_nested_folds
)
from src.models.quantum_vqc import (
    HybridQuantumModel, save_hybrid_checkpoint,
    load_hybrid_checkpoint, get_hybrid_checkpoint_path
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_f1_max_threshold(probs, targets):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 81):
        preds = (probs >= t).astype(float)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def run_inner_optuna(outer_train_emb, outer_train_labels, seed, n_trials):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    opt_tr_idx, opt_val_idx = next(sss.split(outer_train_emb, outer_train_labels))

    X_tr  = torch.tensor(outer_train_emb[opt_tr_idx],     dtype=torch.float32)
    X_val = torch.tensor(outer_train_emb[opt_val_idx],    dtype=torch.float32)
    y_tr  = torch.tensor(outer_train_labels[opt_tr_idx],  dtype=torch.float32)
    y_val = outer_train_labels[opt_val_idx]

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [5, 8, 10])
        epochs     = trial.suggest_categorical("epochs", [40, 60, 80])

        set_all_seeds(seed)
        model = HybridQuantumModel(
            input_dim=cfg.INPUT_DIM,
            n_qubits=cfg.N_QUBITS, n_layers=cfg.N_LAYERS,
            nn_hidden=cfg.NN_HIDDEN, nn_depth=cfg.NN_DEPTH,
            pre_hidden=cfg.PRE_HIDDEN,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn   = nn.BCEWithLogitsLoss()

        eff_bs   = max(2, min(batch_size, len(X_tr) - 1))
        train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=eff_bs, shuffle=True, drop_last=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in train_dl:
                logits = model(xb)
                loss   = loss_fn(logits, yb.unsqueeze(1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_val)).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(float)
        trial.set_user_attr("val_probs", probs.tolist())
        return f1_score(y_val, preds, zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_val_probs = np.array(study.best_trial.user_attrs["val_probs"])
    best_threshold = find_f1_max_threshold(best_val_probs, y_val)
    return study.best_params, best_threshold


def train_one_fold(model, optimizer, loss_fn, train_dl, X_test, y_test, epochs, threshold=0.5):
    train_start = time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            logits = model(xb)
            loss   = loss_fn(logits, yb.unsqueeze(dim=1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    train_latency = time.time() - train_start

    model.eval()
    infer_start = time.time()
    with torch.no_grad():
        logits  = model(X_test)
        probs   = torch.sigmoid(logits).cpu().numpy().flatten()
        preds   = (probs >= threshold).astype(float)
        targets = y_test.cpu().numpy().flatten()
    infer_latency = time.time() - infer_start

    n_params     = sum(p.numel() for p in model.parameters())
    fold_metrics = calculate_metrics(targets, preds, probs)
    fold_metrics["train_latency_s"]  = train_latency
    fold_metrics["infer_latency_ms"] = infer_latency * 1000 / max(len(targets), 1)
    fold_metrics["total_latency_s"]  = train_latency + infer_latency
    fold_metrics["total_mem_mb"]     = 0.0
    fold_metrics["n_params"]         = n_params
    return fold_metrics, targets, preds, probs


def run_quantum_pipeline():
    import psutil, os as _os
    _proc = psutil.Process(_os.getpid())
    pipeline_start = time.time()
    mem_start = _proc.memory_info().rss
    peak_mem  = mem_start

    os.makedirs(cfg.RESULTS_QUANTUM_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_QUANTUM_DIR, exist_ok=True)

    data_sizes = cfg.DATA_SIZES
    seeds      = cfg.SEEDS
    n_outer    = cfg.N_OUTER_FOLDS

    print(f"SPEDIA - Hybrid Quantum VQC")
    print(f"qubits={cfg.N_QUBITS}, layers={cfg.N_LAYERS}")
    print(f"Seeds: {seeds}  |  Data Sizes: {data_sizes}")
    print(f"{'='*60}\n")

    json_path = os.path.join(cfg.RESULTS_QUANTUM_DIR, "quantum_results.json")
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
            fold_shapiro_ps  = []
            fold_thresholds  = []
            best_outer_score = -1.0

            for outer_idx, fold_data in enumerate(nested_folds):
                outer_train_idx = fold_data["outer_train"]
                outer_test_idx  = fold_data["outer_test"]

                print(f"\nOuter Fold {outer_idx+1}/{n_outer}: "
                      f"Train={len(outer_train_idx)}, Test={len(outer_test_idx)}")

                opt_start = time.time()
                best_params, best_threshold = run_inner_optuna(
                    stage_emb[outer_train_idx],
                    stage_labels[outer_train_idx],
                    seed=seed,
                    n_trials=cfg.OPTUNA_INNER_TRIALS_QUANTUM,
                )
                opt_time = time.time() - opt_start
                print(f"Optuna ({cfg.OPTUNA_INNER_TRIALS_QUANTUM} trials, {opt_time:.0f}s): "
                      f"lr={best_params['lr']:.5f}  bs={best_params['batch_size']}  "
                      f"ep={best_params['epochs']}  threshold={best_threshold:.2f}")

                X_out_train = torch.tensor(stage_emb[outer_train_idx],    dtype=torch.float32)
                X_out_test  = torch.tensor(stage_emb[outer_test_idx],     dtype=torch.float32)
                y_out_train = torch.tensor(stage_labels[outer_train_idx], dtype=torch.float32)
                y_out_test  = torch.tensor(stage_labels[outer_test_idx],  dtype=torch.float32)

                eff_bs   = max(2, min(best_params["batch_size"], len(X_out_train) - 1))
                outer_dl = DataLoader(TensorDataset(X_out_train, y_out_train),
                                      batch_size=eff_bs, shuffle=True, drop_last=True)

                set_all_seeds(seed)
                model = HybridQuantumModel(
                    input_dim=cfg.INPUT_DIM,
                    n_qubits=cfg.N_QUBITS, n_layers=cfg.N_LAYERS,
                    nn_hidden=cfg.NN_HIDDEN, nn_depth=cfg.NN_DEPTH,
                    pre_hidden=cfg.PRE_HIDDEN,
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
                loss_fn   = nn.BCEWithLogitsLoss()

                t_start = time.time()
                outer_metrics, targets, preds, outer_probs = train_one_fold(
                    model, optimizer, loss_fn, outer_dl,
                    X_out_test, y_out_test, best_params["epochs"],
                    threshold=best_threshold,
                )
                fold_time = time.time() - t_start

                current_rss = _proc.memory_info().rss
                if current_rss > peak_mem:
                    peak_mem = current_rss
                fold_thresholds.append(best_threshold)

                if outer_metrics["f1"] > best_outer_score:
                    best_outer_score = outer_metrics["f1"]
                    save_hybrid_checkpoint(
                        model, optimizer, best_params["epochs"], "best",
                        data_size, seed, cfg.CHECKPOINT_QUANTUM_DIR
                    )

                for key in stage_metrics:
                    if key in outer_metrics:
                        stage_metrics[key].append(outer_metrics[key])

                fold_ece = calculate_ece(targets, outer_probs)
                print(f"Fold {outer_idx+1} (train {fold_time:.1f}s): "
                      f"Acc={outer_metrics['acc']:.4f}  MCC={outer_metrics['mcc']:.4f}  "
                      f"F1={outer_metrics['f1']:.4f}  AUC={outer_metrics['auc']:.4f}  "
                      f"ECE={fold_ece:.4f}")

                if outer_idx == n_outer - 1:
                    plot_confusion_matrix(targets, preds,
                        f"Quantum CM (Size={data_size}, Seed={seed})",
                        save_path=os.path.join(cfg.RESULTS_QUANTUM_DIR,
                            f"cm_quantum_seed{seed}_size{data_size}.png"))
                    plot_reliability_diagram(targets, outer_probs,
                        f"Quantum Calibration (Size={data_size}, Seed={seed})",
                        save_path=os.path.join(cfg.RESULTS_QUANTUM_DIR,
                            f"reliability_quantum_seed{seed}_size{data_size}.png"))

            stage_time = time.time() - stage_start
            print(f"Stage {data_size} done in {stage_time:.1f}s")

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
    print(f"\nTotal: {int(total_time//3600):02d}h {int((total_time%3600)//60):02d}m "
          f"{int(total_time%60):02d}s | Peak mem: {peak_mem_mb:.1f} MB")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_quantum_pipeline()
