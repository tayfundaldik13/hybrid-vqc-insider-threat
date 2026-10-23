import os
import sys
import time
import random
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.models.transformer import BehaviorTransformer
from src.data_process import load_behavioral_features, create_nonoverlapping_stages, create_nested_folds
from src.utils import (
    calculate_metrics, plot_confusion_matrix,
    calculate_ece, plot_reliability_diagram,
    print_stage_report, print_seed_summary,
    print_final_aggregation_report, save_results_json,
    _load_partial, _save_partial, _clear_partial,
)

RESULTS_DIR    = config.RESULTS_TRANSFORMER_DIR
CHECKPOINT_DIR = config.CHECKPOINT_TRANSFORMER_DIR

INPUT_DIM      = 16
D_MODEL        = config.TF_D_MODEL
N_HEADS        = config.TF_N_HEADS
N_LAYERS       = config.TF_N_LAYERS
DROPOUT        = config.TF_DROPOUT
DIM_FEEDFORWARD = config.TF_DIM_FEEDFORWARD
LR             = config.TF_LR
BATCH_SIZE     = config.TF_BATCH_SIZE
EPOCHS         = config.TF_EPOCHS
WEIGHT_DECAY   = config.TF_WEIGHT_DECAY
PATIENCE       = config.EARLY_STOP_PATIENCE
MIN_DELTA      = config.EARLY_STOP_MIN_DELTA
VAL_FRAC       = config.EARLY_STOP_VAL_FRAC

DATA_SIZES = config.DATA_SIZES
SEEDS      = config.SEEDS
N_OUTER    = config.N_OUTER_FOLDS


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_pipeline():
    import psutil, os as _os
    _proc = psutil.Process(_os.getpid())
    pipeline_start = time.time()
    mem_start = _proc.memory_info().rss
    peak_mem = mem_start

    device = config.DEVICE

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Transformer PIPELINE - Staged 50-500 (r4.2)")
    print(f"   Architecture: PCA(16) -> {N_LAYERS}xTransformerEncoder(d_model={D_MODEL}, heads={N_HEADS})")
    print(f"   LR={LR:.5f}  BatchSize={BATCH_SIZE}  Epochs={EPOCHS}  WD={WEIGHT_DECAY:.2e}  Dropout={DROPOUT:.3f}")
    print(f"   Seeds: {SEEDS}  |  Stages: {DATA_SIZES}  |  Nested CV: {N_OUTER}  |  Device: {device}")
    print(f"{'='*60}\n")

    full_embeddings, full_labels = load_behavioral_features()
    n_total = len(full_labels)
    print(f"   Full dataset: {n_total} samples  "
          f"(threat={int(full_labels.sum())}, "
          f"normal={int((full_labels==0).sum())}, "
          f"ratio={full_labels.mean():.4f})\n")

    json_path = os.path.join(RESULTS_DIR, "transformer_results.json")
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
            nested_folds = create_nested_folds(stage_labels, N_OUTER, n_inner=1, seed=seed)

            stage_metrics = {
                "acc": [], "kappa": [], "mcc": [], "precision": [],
                "recall": [], "specificity": [], "f1": [], "macro_f1": [], "auc": [],
                "auprc": [], "ece": [],
                "train_latency_s": [], "infer_latency_ms": [],
                "total_latency_s": [], "total_mem_mb": [], "n_params": []
            }
            fold_shapiro_ps = []
            best_outer_score = -1.0
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

                X_train_np = stage_emb[outer_train_idx]
                X_test_np  = stage_emb[outer_test_idx]
                y_train_np = stage_labels[outer_train_idx]
                y_test_np  = stage_labels[outer_test_idx]

                pos_weight_val = float((y_train_np == 0).sum() / max((y_train_np == 1).sum(), 1))
                pos_w = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

                X_tr = torch.tensor(X_train_np, dtype=torch.float32).to(device)
                X_te = torch.tensor(X_test_np,  dtype=torch.float32).to(device)
                y_tr = torch.tensor(y_train_np, dtype=torch.float32).to(device)

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
                val_pw = torch.tensor([
                    float((y_val.cpu().numpy() == 0).sum() / max((y_val.cpu().numpy() == 1).sum(), 1))
                ], dtype=torch.float32).to(device)
                val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=val_pw)

                eff_bs2 = max(2, min(BATCH_SIZE, len(X_tr2) - 1))
                train_dl = DataLoader(TensorDataset(X_tr2, y_tr2),
                                      batch_size=eff_bs2, shuffle=True, drop_last=True)

                set_all_seeds(seed)
                model = BehaviorTransformer(INPUT_DIM, D_MODEL, N_HEADS, N_LAYERS,
                                            DROPOUT, DIM_FEEDFORWARD).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_w)

                t0 = time.time()
                best_val = float("inf")
                patience_ct = 0
                best_state = None
                model.train()
                for ep in range(EPOCHS):
                    for xb, yb in train_dl:
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        vl = val_loss_fn(model(X_val), y_val).item()
                    if vl < best_val - MIN_DELTA:
                        best_val = vl
                        patience_ct = 0
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
                    for i in range(0, len(X_te), infer_bs):
                        probs_list.append(torch.sigmoid(model(X_te[i:i+infer_bs])).cpu())
                    probs = torch.cat(probs_list).numpy().flatten()
                infer_latency = time.time() - t1
                preds = (probs >= 0.5).astype(float)

                current_rss = _proc.memory_info().rss
                if current_rss > peak_mem:
                    peak_mem = current_rss
                fold_mem_mb = max(0.0, (current_rss - mem_fold_start) / 1024 / 1024)

                n_params = sum(p.numel() for p in model.parameters())
                fold_metrics = calculate_metrics(y_test_np, preds, probs)
                fold_metrics["ece"]              = calculate_ece(y_test_np, probs)
                fold_metrics["train_latency_s"]  = train_latency
                fold_metrics["infer_latency_ms"] = infer_latency * 1000 / max(len(y_test_np), 1)
                fold_metrics["total_latency_s"]  = train_latency + infer_latency
                fold_metrics["total_mem_mb"]     = fold_mem_mb
                fold_metrics["n_params"]         = n_params

                _pred_dir = os.path.join(RESULTS_DIR, "predictions")
                os.makedirs(_pred_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(_pred_dir, f"seed{seed}_size{data_size}_fold{outer_idx}.npz"),
                    y_true=y_test_np, y_pred=preds, test_idx=outer_test_idx)

                for key in stage_metrics:
                    if key in fold_metrics:
                        stage_metrics[key].append(fold_metrics[key])

                _save_partial(RESULTS_DIR, seed, outer_idx, stage_metrics, stage=data_size)

                if fold_metrics["f1"] > best_outer_score:
                    best_outer_score = fold_metrics["f1"]
                    torch.save({"model_state": model.state_dict()},
                               os.path.join(CHECKPOINT_DIR, f"transformer_seed{seed}_size{data_size}_best.pt"))

                print(f"  >>> Fold {outer_idx+1}({train_latency:.1f}s): "
                      f"Acc={fold_metrics['acc']:.4f} MCC={fold_metrics['mcc']:.4f} "
                      f"F1={fold_metrics['f1']:.4f} AUC={fold_metrics['auc']:.4f}")

                if outer_idx == N_OUTER - 1:
                    plot_confusion_matrix(y_test_np, preds,
                        f"Transformer CM (Size={data_size}, Seed={seed})",
                        save_path=os.path.join(RESULTS_DIR, f"cm_transformer_seed{seed}_size{data_size}.png"))
                    plot_reliability_diagram(y_test_np, probs,
                        f"Transformer Reliability (Size={data_size}, Seed={seed})",
                        save_path=os.path.join(RESULTS_DIR, f"reliability_transformer_seed{seed}_size{data_size}.png"))

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
          f"  Peak memory: {peak_mem_mb:.1f}MB\n  Results: {json_path}\n{'='*60}\nTransformer (staged) done!")


if __name__ == "__main__":
    run_pipeline()
