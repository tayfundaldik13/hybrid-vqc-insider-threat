import os
import sys
import time
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from scipy import stats as sp_stats

_ALL_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_ALL_DATA_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _ALL_DATA_DIR)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    os.system("pip install optuna -q")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

import importlib.util as _ilu
_cfg = _ilu.spec_from_file_location("src.config", os.path.join(_ALL_DATA_DIR, "src", "config.py"))
config = _ilu.module_from_spec(_cfg); _cfg.loader.exec_module(config)
_qcfg = _ilu.spec_from_file_location("src.quantum_config", os.path.join(_ROOT_DIR, "src", "quantum_config.py"))
qconfig = _ilu.module_from_spec(_qcfg); _qcfg.loader.exec_module(qconfig)
del _ilu, _cfg, _qcfg
from src.models.quantum_vqc import (
    HybridQuantumModel, save_hybrid_checkpoint,
    get_hybrid_checkpoint_path
)
from src.data_process import load_behavioral_features, create_nested_folds
from src.utils import (
    calculate_metrics, plot_confusion_matrix,
    calculate_ece, plot_reliability_diagram,
    print_stage_report, print_seed_summary,
    print_final_aggregation_report, save_results_json,
    _load_partial, _save_partial, _clear_partial
)

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_all", "quantum")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_all", "quantum")

STAGE_KEY = "all"


def find_f1_max_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 81):
        preds = (probs >= t).astype(float)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_inner_optuna(outer_train_emb, outer_train_labels, seed, n_trials):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    opt_tr_idx, opt_val_idx = next(sss.split(outer_train_emb, outer_train_labels))

    X_tr  = torch.tensor(outer_train_emb[opt_tr_idx],     dtype=torch.float32)
    X_val = torch.tensor(outer_train_emb[opt_val_idx],    dtype=torch.float32)
    y_tr  = torch.tensor(outer_train_labels[opt_tr_idx],  dtype=torch.float32)
    y_val = torch.tensor(outer_train_labels[opt_val_idx], dtype=torch.float32)

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [5, 8, 10, 16, 32])
        epochs     = trial.suggest_categorical("epochs", [40, 60, 80])

        set_all_seeds(seed)
        model = HybridQuantumModel(
            input_dim=qconfig.INPUT_DIM,
            n_qubits=qconfig.N_QUBITS,
            n_layers=qconfig.N_LAYERS,
            nn_hidden=qconfig.NN_HIDDEN,
            nn_depth=qconfig.NN_DEPTH,
            pre_hidden=qconfig.PRE_HIDDEN,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn   = nn.BCEWithLogitsLoss()

        eff_bs = max(2, min(batch_size, len(X_tr) - 1))
        train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=eff_bs, shuffle=True, drop_last=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in train_dl:
                logits = model(xb)
                loss   = loss_fn(logits, yb.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_val)).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(float)
        trial.set_user_attr("val_probs", probs.tolist())
        return f1_score(y_val.numpy(), preds, zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_val_probs = np.array(study.best_trial.user_attrs["val_probs"])
    best_threshold = find_f1_max_threshold(best_val_probs, y_val.numpy())
    return study.best_params, best_threshold


def train_one_fold(model, optimizer, loss_fn, train_dl, X_test, y_test,
                   epochs, threshold=0.5,
                   X_val=None, y_val=None, val_loss_fn=None,
                   patience=20, min_delta=1e-4):
    import copy
    train_start = time.time()
    best_vl = float("inf"); pat_ct = 0; best_st = None

    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            logits = model(xb)
            loss   = loss_fn(logits, yb.unsqueeze(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vl = val_loss_fn(model(X_val), y_val.unsqueeze(1)).item()
            if vl < best_vl - min_delta:
                best_vl = vl; pat_ct = 0; best_st = copy.deepcopy(model.state_dict())
            else:
                pat_ct += 1
                if pat_ct >= patience:
                    print(f"    Quantum early stop ep {ep+1}/{epochs} val_loss={best_vl:.4f}")
                    break

    if best_st is not None:
        model.load_state_dict(best_st)
    train_latency = time.time() - train_start

    model.eval()
    infer_start = time.time()
    with torch.no_grad():
        logits  = model(X_test)
        probs   = torch.sigmoid(logits).cpu().numpy().flatten()
        preds   = (probs >= threshold).astype(float)
        targets = y_test.cpu().numpy().flatten()
    infer_latency = time.time() - infer_start

    n_params = sum(p.numel() for p in model.parameters())
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

    seeds   = config.SEEDS
    n_outer = config.N_OUTER_FOLDS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"HYBRID QUANTUM VQC PIPELINE - FULL DATASET")
    print(f"   Architecture: Behavioral(16-dim PCA) -> Pre-Quantum(16->{2**qconfig.N_QUBITS}) -> "
          f"AmplitudeEmbedding -> StronglyEntanglingLayers -> Post-Quantum")
    print(f"   VQC: qubits={qconfig.N_QUBITS}, layers={qconfig.N_LAYERS}")
    print(f"   Post-Quantum NN: hidden={qconfig.NN_HIDDEN}, depth={qconfig.NN_DEPTH}")
    print(f"   Inner Optuna: {qconfig.OPTUNA_INNER_TRIALS} trials x 75/25 split")
    print(f"   Seeds: {seeds}  |  Nested CV: {n_outer} outer folds")
    print(f"{'='*60}\n")

    full_embeddings, full_labels = load_behavioral_features()
    n_total = len(full_labels)
    print(f"   Full dataset: {n_total} samples  "
          f"(threat={int(full_labels.sum())}, "
          f"normal={int((full_labels==0).sum())}, "
          f"ratio={full_labels.mean():.4f})\n")

    json_path = os.path.join(RESULTS_DIR, "quantum_all_results.json")
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
            "ece": [],
            "train_latency_s": [], "infer_latency_ms": [],
            "total_latency_s": [], "total_mem_mb": [], "n_params": []
        }
        fold_shapiro_ps  = []
        fold_thresholds  = []
        best_outer_score = -1.0
        stage_start      = time.time()

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

            opt_start = time.time()
            best_params, best_threshold = run_inner_optuna(
                full_embeddings[outer_train_idx],
                full_labels[outer_train_idx],
                seed=seed,
                n_trials=qconfig.OPTUNA_INNER_TRIALS,
            )
            opt_time = time.time() - opt_start
            print(f"     Optuna ({qconfig.OPTUNA_INNER_TRIALS} trials, {opt_time:.0f}s): "
                  f"lr={best_params['lr']:.5f}  "
                  f"bs={best_params['batch_size']}  "
                  f"ep={best_params['epochs']}  "
                  f"threshold={best_threshold:.2f}")

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
            qval_idx = np.concatenate([rng_q.choice(pos_i, min(n_vp,len(pos_i)), replace=False),
                                        rng_q.choice(neg_i, min(n_vn,len(neg_i)), replace=False)])
            qtr_idx  = np.setdiff1d(np.arange(len(y_otr_np)), qval_idx)
            X_qval = X_out_train[qval_idx]; y_qval = y_out_train[qval_idx]
            X_qtr  = X_out_train[qtr_idx];  y_qtr  = y_out_train[qtr_idx]
            val_pw_q = torch.tensor([float((y_qval.numpy()==0).sum()/
                                           max((y_qval.numpy()==1).sum(),1))],
                                    dtype=torch.float32)
            q_val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=val_pw_q)
            eff_bs2 = max(2, min(eff_bs, len(X_qtr) - 1))
            outer_dl = DataLoader(
                TensorDataset(X_qtr, y_qtr),
                batch_size=eff_bs2, shuffle=True, drop_last=True
            )

            set_all_seeds(seed)
            model = HybridQuantumModel(
                input_dim=qconfig.INPUT_DIM,
                n_qubits=qconfig.N_QUBITS,
                n_layers=qconfig.N_LAYERS,
                nn_hidden=qconfig.NN_HIDDEN,
                nn_depth=qconfig.NN_DEPTH,
                pre_hidden=qconfig.PRE_HIDDEN,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
            loss_fn   = nn.BCEWithLogitsLoss()

            t_start = time.time()
            outer_metrics, targets, preds, outer_probs = train_one_fold(
                model, optimizer, loss_fn, outer_dl,
                X_out_test, y_out_test, best_params["epochs"],
                threshold=best_threshold,
                X_val=X_qval, y_val=y_qval, val_loss_fn=q_val_loss_fn,
                patience=config.EARLY_STOP_PATIENCE,
                min_delta=config.EARLY_STOP_MIN_DELTA,
            )
            fold_time = time.time() - t_start

            current_rss = _proc.memory_info().rss
            if current_rss > peak_mem:
                peak_mem = current_rss
            outer_metrics["total_mem_mb"] = max(0.0, (current_rss - mem_fold_start) / 1024 / 1024)
            fold_thresholds.append(best_threshold)

            if outer_metrics["f1"] > best_outer_score:
                best_outer_score = outer_metrics["f1"]
                save_hybrid_checkpoint(
                    model, optimizer, best_params["epochs"], "best",
                    STAGE_KEY, seed, CHECKPOINT_DIR
                )

            outer_metrics["ece"] = calculate_ece(targets, outer_probs)

            _pred_dir = os.path.join(RESULTS_DIR, "predictions")
            os.makedirs(_pred_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(_pred_dir, f"seed{seed}_fold{outer_idx}.npz"),
                y_true=targets, y_pred=preds, test_idx=outer_test_idx)
            for key in stage_metrics:
                if key in outer_metrics:
                    stage_metrics[key].append(outer_metrics[key])

            _save_partial(RESULTS_DIR, seed, outer_idx, stage_metrics)

            print(f"  >>> Outer {outer_idx+1} (train {fold_time:.1f}s): "
                  f"Acc={outer_metrics['acc']:.4f}  "
                  f"MCC={outer_metrics['mcc']:.4f}  "
                  f"F1={outer_metrics['f1']:.4f}  "
                  f"MacroF1={outer_metrics['macro_f1']:.4f}  "
                  f"AUC={outer_metrics['auc']:.4f}  "
                  f"ECE={outer_metrics['ece']:.4f}")

            if outer_idx == n_outer - 1:
                cm_path = os.path.join(RESULTS_DIR, f"cm_quantum_all_seed{seed}.png")
                plot_confusion_matrix(targets, preds,
                    f"Quantum CM (Full Data, Seed={seed})", save_path=cm_path)
                rel_path = os.path.join(RESULTS_DIR, f"reliability_quantum_all_seed{seed}.png")
                plot_reliability_diagram(targets, outer_probs,
                    f"Quantum Calibration (Full Data, Seed={seed})", save_path=rel_path)

        stage_time = time.time() - stage_start
        print(f"\n   Seed {seed} completed in {stage_time:.1f}s")

        f1_scores = stage_metrics.get("f1", [])
        if len(f1_scores) >= 3:
            _, p_sh = sp_stats.shapiro(f1_scores)
            fold_shapiro_ps.append(p_sh)

        mean_threshold = float(np.mean(fold_thresholds)) if fold_thresholds else 0.5
        print(f"   Mean F1-max threshold (folds): {mean_threshold:.3f}")

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
    print(f"  Total Pipeline Latency : "
          f"{int(total_time//3600):02d}h "
          f"{int((total_time%3600)//60):02d}m "
          f"{int(total_time%60):02d}s")
    print(f"  Peak Memory Usage      : {peak_mem_mb:.1f} MB")
    print(f"  Results saved to       : {json_path}")
    print(f"{'='*60}")
    print("\nHybrid Quantum VQC (Full Data) Pipeline completed!")


if __name__ == "__main__":
    run_quantum_pipeline()
