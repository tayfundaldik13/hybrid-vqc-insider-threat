import os
import sys
import time
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
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

RESULTS_DIR    = os.path.join(_ALL_DATA_DIR, "results_all", "mlp")
CHECKPOINT_DIR = os.path.join(_ALL_DATA_DIR, "checkpoints_all", "mlp")

INPUT_DIM    = 16

HIDDEN_SIZE  = config.MLP_HIDDEN_SIZE
NUM_LAYERS   = config.MLP_NUM_LAYERS
DROPOUT      = config.MLP_DROPOUT
LR           = config.MLP_LR
BATCH_SIZE   = config.MLP_BATCH_SIZE
EPOCHS       = config.MLP_EPOCHS
WEIGHT_DECAY = config.MLP_WEIGHT_DECAY
PATIENCE    = config.EARLY_STOP_PATIENCE
MIN_DELTA   = config.EARLY_STOP_MIN_DELTA
VAL_FRAC    = config.EARLY_STOP_VAL_FRAC
STAGE_KEY    = "all"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_pipeline():
    import psutil, os as _os
    _proc = psutil.Process(_os.getpid())
    pipeline_start = time.time()
    mem_start = _proc.memory_info().rss
    peak_mem  = mem_start

    device  = config.DEVICE
    seeds   = config.SEEDS
    n_outer = config.N_OUTER_FOLDS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("MLP PIPELINE - FULL DATASET")
    print(f"   Architecture: PCA(16) -> {NUM_LAYERS}x[Linear({HIDDEN_SIZE})->LN->ReLU->Drop] -> Linear(1)")
    print(f"   LR={LR:.5f}  BatchSize={BATCH_SIZE}  Epochs={EPOCHS}  "
          f"WD={WEIGHT_DECAY:.2e}  Dropout={DROPOUT:.3f}")
    print(f"   Seeds: {seeds}  |  Nested CV: {n_outer} outer folds  |  Device: {device}")
    print(f"{'='*60}\n")

    full_embeddings, full_labels = load_behavioral_features()
    n_total = len(full_labels)
    print(f"   Full dataset: {n_total} samples  "
          f"(threat={int(full_labels.sum())}, "
          f"normal={int((full_labels==0).sum())}, "
          f"ratio={full_labels.mean():.4f})\n")

    json_path = os.path.join(RESULTS_DIR, "mlp_all_results.json")
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
        fold_shapiro_ps  = []
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

            X_train_np = full_embeddings[outer_train_idx]
            X_test_np  = full_embeddings[outer_test_idx]
            y_train_np = full_labels[outer_train_idx]
            y_test_np  = full_labels[outer_test_idx]

            pos_weight_val = float(
                (y_train_np == 0).sum() / max((y_train_np == 1).sum(), 1))
            pos_w = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

            X_tr = torch.tensor(X_train_np, dtype=torch.float32).to(device)
            X_vl = torch.tensor(X_test_np,  dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_train_np, dtype=torch.float32).to(device)

            set_all_seeds(seed)
            model     = SimpleMLP(INPUT_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                          weight_decay=WEIGHT_DECAY)
            loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_w)

            eff_bs   = max(2, min(BATCH_SIZE, len(X_tr) - 1))
            train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                                  batch_size=eff_bs, shuffle=True, drop_last=True)

            t0 = time.time()
            import copy
            pos_tr = np.where(y_train_np == 1)[0]
            neg_tr = np.where(y_train_np == 0)[0]
            n_val  = max(2, int(len(y_train_np) * VAL_FRAC))
            n_vp   = max(1, int(n_val * y_train_np.mean()))
            n_vn   = n_val - n_vp
            rng    = np.random.default_rng(seed)
            val_pos = rng.choice(pos_tr, min(n_vp, len(pos_tr)), replace=False)
            val_neg = rng.choice(neg_tr, min(n_vn, len(neg_tr)), replace=False)
            val_idx = np.concatenate([val_pos, val_neg])
            tr_idx  = np.setdiff1d(np.arange(len(y_train_np)), val_idx)

            X_tr2 = X_tr[tr_idx];  y_tr2 = y_tr[tr_idx]
            X_val = X_tr[val_idx]; y_val = y_tr[val_idx]
            val_pw = torch.tensor(
                [float((y_val.cpu().numpy()==0).sum() / max((y_val.cpu().numpy()==1).sum(),1))],
                dtype=torch.float32).to(device)
            val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=val_pw)

            eff_bs2  = max(2, min(BATCH_SIZE, len(X_tr2) - 1))
            train_dl = DataLoader(TensorDataset(X_tr2, y_tr2),
                                  batch_size=eff_bs2, shuffle=True, drop_last=True)

            best_val   = float("inf"); patience_ct = 0; best_state = None
            model.train()
            for ep in range(EPOCHS):
                for xb, yb in train_dl:
                    logits = model(xb)
                    loss   = loss_fn(logits, yb)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    vl = val_loss_fn(model(X_val), y_val).item()
                if vl < best_val - MIN_DELTA:
                    best_val = vl; patience_ct = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience_ct += 1
                    if patience_ct >= PATIENCE:
                        print(f"    Early stop ep {ep+1}/{EPOCHS} val_loss={best_val:.4f}")
                        break
                model.train()
            if best_state is not None:
                model.load_state_dict(best_state)
            train_latency = time.time() - t0

            t1 = time.time()
            model.eval()
            with torch.no_grad():
                infer_bs = max(256, BATCH_SIZE)
                probs_list = []
                for i in range(0, len(X_vl), infer_bs):
                    probs_list.append(torch.sigmoid(model(X_vl[i:i+infer_bs])).cpu())
                probs = torch.cat(probs_list).numpy()
            infer_latency = time.time() - t1
            preds = (probs >= 0.5).astype(float)

            current_rss = _proc.memory_info().rss
            if current_rss > peak_mem:
                peak_mem = current_rss
            fold_mem_mb = max(0.0, (current_rss - mem_fold_start) / 1024 / 1024)

            n_params     = sum(p.numel() for p in model.parameters())
            fold_metrics = calculate_metrics(y_test_np, preds, probs)
            fold_metrics["train_latency_s"]  = train_latency
            fold_metrics["infer_latency_ms"] = infer_latency * 1000 / max(len(y_test_np), 1)
            fold_metrics["total_latency_s"]  = train_latency + infer_latency
            fold_metrics["total_mem_mb"]     = fold_mem_mb
            fold_metrics["n_params"]         = n_params
            fold_metrics["ece"]              = calculate_ece(y_test_np, probs)

            _pred_dir = os.path.join(RESULTS_DIR, "predictions")
            os.makedirs(_pred_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(_pred_dir, f"seed{seed}_fold{outer_idx}.npz"),
                y_true=y_test_np, y_pred=preds, test_idx=outer_test_idx)

            for key in stage_metrics:
                if key in fold_metrics:
                    stage_metrics[key].append(fold_metrics[key])

            _save_partial(RESULTS_DIR, seed, outer_idx, stage_metrics)

            if fold_metrics["f1"] > best_outer_score:
                best_outer_score = fold_metrics["f1"]
                ckpt_path = os.path.join(CHECKPOINT_DIR,
                                         f"mlp_all_seed{seed}_best.pt")
                torch.save({"model_state": model.state_dict(),
                            "epoch": EPOCHS}, ckpt_path)

            print(f"  >>> Outer {outer_idx+1} ({train_latency:.1f}s): "
                  f"Acc={fold_metrics['acc']:.4f}  MCC={fold_metrics['mcc']:.4f}  "
                  f"F1={fold_metrics['f1']:.4f}  AUC={fold_metrics['auc']:.4f}")

            if outer_idx == n_outer - 1:
                cm_path = os.path.join(RESULTS_DIR, f"cm_mlp_all_seed{seed}.png")
                plot_confusion_matrix(y_test_np, preds,
                    f"MLP CM (Full Data, Seed={seed})", save_path=cm_path)
                rel_path = os.path.join(RESULTS_DIR, f"reliability_mlp_all_seed{seed}.png")
                plot_reliability_diagram(y_test_np, probs,
                    f"MLP Calibration (Full Data, Seed={seed})", save_path=rel_path)

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
    print("MLP (Full Data) Pipeline completed!")


if __name__ == "__main__":
    run_pipeline()
