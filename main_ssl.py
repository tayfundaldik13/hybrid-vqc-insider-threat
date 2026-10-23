import os
import sys
import time
import random
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.models.ssl import SCARFEncoder, SSLClassifier, corrupt_features, nt_xent_loss
from src.data_process import load_behavioral_features, create_nonoverlapping_stages, create_nested_folds
from src.utils import (
    calculate_metrics, plot_confusion_matrix,
    calculate_ece, plot_reliability_diagram,
    print_stage_report, print_seed_summary,
    print_final_aggregation_report, save_results_json,
    _load_partial, _save_partial, _clear_partial,
)

RESULTS_DIR    = config.RESULTS_SSL_DIR
CHECKPOINT_DIR = config.CHECKPOINT_SSL_DIR

INPUT_DIM        = 16
HIDDEN_DIM       = config.SSL_HIDDEN_DIM
PROJ_DIM         = config.SSL_PROJ_DIM
N_ENCODER_LAYERS = config.SSL_N_ENCODER_LAYERS
DROPOUT          = config.SSL_DROPOUT
CORRUPTION_RATE  = config.SSL_CORRUPTION_RATE
TEMPERATURE      = config.SSL_TEMPERATURE
PRETRAIN_LR      = config.SSL_PRETRAIN_LR
PRETRAIN_EPOCHS  = config.SSL_PRETRAIN_EPOCHS
FINETUNE_LR      = config.SSL_FINETUNE_LR
FINETUNE_EPOCHS  = config.SSL_FINETUNE_EPOCHS
BATCH_SIZE       = config.SSL_BATCH_SIZE
FREEZE_ENCODER   = config.SSL_FREEZE_ENCODER
PATIENCE         = config.EARLY_STOP_PATIENCE
MIN_DELTA        = config.EARLY_STOP_MIN_DELTA
VAL_FRAC         = config.EARLY_STOP_VAL_FRAC

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

    print("Self-Supervised (SCARF) PIPELINE - Staged 50-500 (r4.2)")
    print(f"   Encoder: {N_ENCODER_LAYERS}xLinear({HIDDEN_DIM})+BN+ReLU  Proj:{HIDDEN_DIM}->{PROJ_DIM}")
    print(f"   Corruption={CORRUPTION_RATE:.3f}  Temperature={TEMPERATURE:.3f}")
    print(f"   Pretrain: lr={PRETRAIN_LR:.2e} ep={PRETRAIN_EPOCHS}  "
          f"Finetune: lr={FINETUNE_LR:.2e} ep={FINETUNE_EPOCHS}  FreezeEnc:{FREEZE_ENCODER}")
    print(f"   Seeds: {SEEDS}  |  Stages: {DATA_SIZES}  |  Nested CV: {N_OUTER}  |  Device: {device}")
    print(f"{'='*60}\n")

    full_embeddings, full_labels = load_behavioral_features()
    n_total = len(full_labels)
    print(f"   Full dataset: {n_total} samples  "
          f"(threat={int(full_labels.sum())}, "
          f"normal={int((full_labels==0).sum())}, "
          f"ratio={full_labels.mean():.4f})\n")

    json_path = os.path.join(RESULTS_DIR, "ssl_results.json")
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

                X_tr = torch.tensor(X_train_np, dtype=torch.float32).to(device)
                X_te = torch.tensor(X_test_np,  dtype=torch.float32).to(device)
                y_tr = torch.tensor(y_train_np, dtype=torch.float32).to(device)
                marginals = X_tr.cpu()

                pos_weight_val = float((y_train_np == 0).sum() / max((y_train_np == 1).sum(), 1))
                pos_w = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

                eff_bs = max(4, min(BATCH_SIZE, len(X_tr) - 1))

                set_all_seeds(seed)
                encoder = SCARFEncoder(INPUT_DIM, HIDDEN_DIM, PROJ_DIM,
                                        N_ENCODER_LAYERS, DROPOUT).to(device)
                opt_pre = torch.optim.Adam(encoder.parameters(), lr=PRETRAIN_LR)

                n_pre_val = max(2, int(len(X_tr) * VAL_FRAC))
                pre_val_idx = np.random.choice(len(X_tr), n_pre_val, replace=False)
                pre_tr_idx  = np.setdiff1d(np.arange(len(X_tr)), pre_val_idx)
                X_ptr = X_tr[pre_tr_idx]
                X_ptv = X_tr[pre_val_idx]
                dl_pre = DataLoader(TensorDataset(X_ptr), batch_size=eff_bs,
                                    shuffle=True, drop_last=True)

                t0 = time.time()
                encoder.train()
                best_pre = float("inf")
                patience_pre = 0
                best_enc = None
                for ep in range(PRETRAIN_EPOCHS):
                    for (xb,) in dl_pre:
                        x_c = corrupt_features(xb, CORRUPTION_RATE, marginals).to(device)
                        loss = nt_xent_loss(encoder.project(xb), encoder.project(x_c), TEMPERATURE)
                        opt_pre.zero_grad()
                        loss.backward()
                        opt_pre.step()
                    encoder.eval()
                    with torch.no_grad():
                        x_cv = corrupt_features(X_ptv, CORRUPTION_RATE, marginals).to(device)
                        vl_pre = nt_xent_loss(encoder.project(X_ptv), encoder.project(x_cv), TEMPERATURE).item()
                    if vl_pre < best_pre - MIN_DELTA:
                        best_pre = vl_pre
                        patience_pre = 0
                        best_enc = copy.deepcopy(encoder.state_dict())
                    else:
                        patience_pre += 1
                        if patience_pre >= PATIENCE:
                            print(f"    Pretrain early stop ep {ep+1}/{PRETRAIN_EPOCHS}")
                            break
                    encoder.train()
                if best_enc is not None:
                    encoder.load_state_dict(best_enc)

                if FREEZE_ENCODER:
                    for p in encoder.encoder.parameters():
                        p.requires_grad = False
                clf = SSLClassifier(encoder, HIDDEN_DIM).to(device)
                opt_ft = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, clf.parameters()), lr=FINETUNE_LR)

                pos_tr = np.where(y_train_np == 1)[0]
                neg_tr = np.where(y_train_np == 0)[0]
                n_ftv  = max(2, int(len(y_train_np) * VAL_FRAC))
                n_vp   = max(1, int(n_ftv * y_train_np.mean()))
                n_vn   = n_ftv - n_vp
                rng    = np.random.default_rng(seed)
                ft_val_pos = rng.choice(pos_tr, min(n_vp, len(pos_tr)), replace=False)
                ft_val_neg = rng.choice(neg_tr, min(n_vn, len(neg_tr)), replace=False)
                ft_val_idx = np.concatenate([ft_val_pos, ft_val_neg])
                ft_tr_idx  = np.setdiff1d(np.arange(len(y_train_np)), ft_val_idx)

                X_ftv = X_tr[ft_val_idx]; y_ftv = y_tr[ft_val_idx]
                X_ftt = X_tr[ft_tr_idx];  y_ftt = y_tr[ft_tr_idx]
                val_pw = torch.tensor([
                    float((y_ftv.cpu().numpy() == 0).sum() / max((y_ftv.cpu().numpy() == 1).sum(), 1))
                ], dtype=torch.float32).to(device)
                val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=val_pw)

                eff_ft = max(2, min(BATCH_SIZE, len(X_ftt) - 1))
                dl_ft = DataLoader(TensorDataset(X_ftt, y_ftt), batch_size=eff_ft,
                                   shuffle=True, drop_last=True)

                clf.train()
                best_ft = float("inf")
                patience_ft = 0
                best_clf = None
                for ep in range(FINETUNE_EPOCHS):
                    for xb, yb in dl_ft:
                        opt_ft.zero_grad()
                        loss = loss_fn(clf(xb), yb)
                        loss.backward()
                        opt_ft.step()
                    clf.eval()
                    with torch.no_grad():
                        vl_ft = val_loss_fn(clf(X_ftv), y_ftv).item()
                    if vl_ft < best_ft - MIN_DELTA:
                        best_ft = vl_ft
                        patience_ft = 0
                        best_clf = copy.deepcopy(clf.state_dict())
                    else:
                        patience_ft += 1
                        if patience_ft >= PATIENCE:
                            print(f"    Finetune early stop ep {ep+1}/{FINETUNE_EPOCHS}")
                            break
                    clf.train()
                if best_clf is not None:
                    clf.load_state_dict(best_clf)
                train_latency = time.time() - t0

                t1 = time.time()
                clf.eval()
                with torch.no_grad():
                    infer_bs = max(256, BATCH_SIZE)
                    probs_list = []
                    for i in range(0, len(X_te), infer_bs):
                        probs_list.append(torch.sigmoid(clf(X_te[i:i+infer_bs])).cpu())
                    probs = torch.cat(probs_list).numpy().flatten()
                infer_latency = time.time() - t1
                preds = (probs >= 0.5).astype(float)

                current_rss = _proc.memory_info().rss
                if current_rss > peak_mem:
                    peak_mem = current_rss
                fold_mem_mb = max(0.0, (current_rss - mem_fold_start) / 1024 / 1024)

                n_params = sum(p.numel() for p in clf.parameters())
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
                    torch.save({"model_state": clf.state_dict()},
                               os.path.join(CHECKPOINT_DIR, f"ssl_seed{seed}_size{data_size}_best.pt"))

                print(f"  >>> Fold {outer_idx+1}({train_latency:.1f}s): "
                      f"Acc={fold_metrics['acc']:.4f} MCC={fold_metrics['mcc']:.4f} "
                      f"F1={fold_metrics['f1']:.4f} AUC={fold_metrics['auc']:.4f}")

                if outer_idx == N_OUTER - 1:
                    plot_confusion_matrix(y_test_np, preds,
                        f"SSL-SCARF CM (Size={data_size}, Seed={seed})",
                        save_path=os.path.join(RESULTS_DIR, f"cm_ssl_seed{seed}_size{data_size}.png"))
                    plot_reliability_diagram(y_test_np, probs,
                        f"SSL-SCARF Reliability (Size={data_size}, Seed={seed})",
                        save_path=os.path.join(RESULTS_DIR, f"reliability_ssl_seed{seed}_size{data_size}.png"))

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
          f"  Peak memory: {peak_mem_mb:.1f}MB\n  Results: {json_path}\n{'='*60}\nSSL-SCARF (staged) done!")


if __name__ == "__main__":
    run_pipeline()
