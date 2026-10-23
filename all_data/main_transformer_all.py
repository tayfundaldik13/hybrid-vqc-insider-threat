import os, sys, time, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats as sp_stats

_ALL_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_ALL_DATA_DIR)
sys.path.insert(0, _ROOT_DIR); sys.path.insert(0, _ALL_DATA_DIR)

import importlib.util as _ilu
_cfg = _ilu.spec_from_file_location("src.config", os.path.join(_ALL_DATA_DIR, "src", "config.py"))
config = _ilu.module_from_spec(_cfg); _cfg.loader.exec_module(config)
del _ilu, _cfg
from src.data_process import load_behavioral_features, create_nested_folds
from src.utils import (calculate_metrics, plot_confusion_matrix, calculate_ece,
    plot_reliability_diagram, print_stage_report, print_final_aggregation_report,
    save_results_json, _load_partial, _save_partial, _clear_partial)

RESULTS_DIR    = os.path.join(_ALL_DATA_DIR, "results_all", "transformer")
CHECKPOINT_DIR = os.path.join(_ALL_DATA_DIR, "checkpoints_all", "transformer")
INPUT_DIM      = 16

D_MODEL         = config.TF_D_MODEL
N_HEADS         = config.TF_N_HEADS
N_LAYERS        = config.TF_N_LAYERS
DROPOUT         = config.TF_DROPOUT
DIM_FEEDFORWARD = config.TF_DIM_FEEDFORWARD
LR              = config.TF_LR
BATCH_SIZE      = config.TF_BATCH_SIZE
EPOCHS          = config.TF_EPOCHS
WEIGHT_DECAY    = config.TF_WEIGHT_DECAY
STAGE_KEY    = "all"
PATIENCE     = config.EARLY_STOP_PATIENCE
MIN_DELTA    = config.EARLY_STOP_MIN_DELTA
VAL_FRAC     = config.EARLY_STOP_VAL_FRAC


def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class BehaviorTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout, dim_feedforward):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout  = nn.Dropout(dropout)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x) + self.pos_emb
        x = self.encoder(x).mean(dim=1)
        return self.head(self.dropout(x)).squeeze(-1)


def run_pipeline():
    import psutil, os as _os
    _proc = psutil.Process(_os.getpid()); t0 = time.time()
    mem0 = _proc.memory_info().rss; peak = mem0
    device = config.DEVICE; seeds = config.SEEDS; n_outer = config.N_OUTER_FOLDS

    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Transformer PIPELINE - FULL DATASET")
    print(f"   d_model={D_MODEL} heads={N_HEADS} layers={N_LAYERS} "
          f"ff={DIM_FEEDFORWARD} dropout={DROPOUT:.3f}")
    print(f"   LR={LR:.5f} bs={BATCH_SIZE} epochs={EPOCHS} wd={WEIGHT_DECAY:.2e}")
    print(f"   Seeds:{seeds}  |  Nested CV:{n_outer}  |  Device:{device}\n{'='*60}\n")

    full_emb, full_lab = load_behavioral_features()
    print(f"   Full dataset:{len(full_lab)} samples "
          f"(threat={int(full_lab.sum())},ratio={full_lab.mean():.4f})\n")

    jp = os.path.join(RESULTS_DIR, "transformer_all_results.json")
    if os.path.exists(jp):
        import json as _j
        with open(jp) as _f: all_results = {int(k): v for k, v in _j.load(_f).items()}
        print(f"Loaded seeds:{sorted(all_results.keys())}")
    else: all_results = {}

    for si, seed in enumerate(seeds):
        if seed in all_results: print(f"\nSEED {seed} done-skip"); continue
        print(f"\n{'#'*60}\nSEED {seed} ({si+1}/{len(seeds)})\n{'#'*60}")
        set_all_seeds(seed)
        folds = create_nested_folds(full_lab, n_outer, n_inner=1, seed=seed)
        sm = {"acc":[],"kappa":[],"mcc":[],"precision":[],"recall":[],"specificity":[],
              "f1":[],"macro_f1":[],"auc":[],"ece":[],"train_latency_s":[],
              "infer_latency_ms":[],"total_latency_s":[],"total_mem_mb":[],"n_params":[]}
        shaps = []; best_s = -1.0; t_stage = time.time()

        _partial = _load_partial(RESULTS_DIR, seed)
        _start_fold = 0
        if _partial is not None:
            _start_fold = _partial["next_fold"]
            _pm = _partial["stage_metrics"]
            for _k in sm:
                if _k in _pm:
                    sm[_k] = list(_pm[_k])
            print(f"  [RESUME] Seed {seed}: resuming from fold {_start_fold+1}")

        for oi, fd in enumerate(folds):
            tri = fd["outer_train"]; tei = fd["outer_test"]
            if oi < _start_fold:
                continue
            print(f"\n  Outer Fold {oi+1}/{n_outer}: Train={len(tri)}, Test={len(tei)}")
            mem_fold_start = _proc.memory_info().rss
            X_tr_np = full_emb[tri]; X_te_np = full_emb[tei]
            y_tr_np = full_lab[tri]; y_te_np = full_lab[tei]

            pw = torch.tensor([float((y_tr_np==0).sum()/max((y_tr_np==1).sum(),1))],
                               dtype=torch.float32).to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

            X_tr = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
            X_te = torch.tensor(X_te_np, dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_tr_np, dtype=torch.float32).to(device)

            set_all_seeds(seed)
            model = BehaviorTransformer(INPUT_DIM, D_MODEL, N_HEADS, N_LAYERS,
                                         DROPOUT, DIM_FEEDFORWARD).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            eff_bs = max(2, min(BATCH_SIZE, len(X_tr)-1))
            dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=eff_bs,
                             shuffle=True, drop_last=True)

            tt = time.time()
            import copy
            pos_tr = np.where(y_tr_np == 1)[0]; neg_tr = np.where(y_tr_np == 0)[0]
            n_val  = max(2, int(len(y_tr_np) * VAL_FRAC))
            n_vp   = max(1, int(n_val * y_tr_np.mean())); n_vn = n_val - n_vp
            rng    = np.random.default_rng(seed)
            val_idx = np.concatenate([rng.choice(pos_tr, min(n_vp,len(pos_tr)), replace=False),
                                      rng.choice(neg_tr, min(n_vn,len(neg_tr)), replace=False)])
            tr_idx2 = np.setdiff1d(np.arange(len(y_tr_np)), val_idx)
            X_tr2 = X_tr[tr_idx2]; y_tr2 = y_tr[tr_idx2]
            X_val = X_tr[val_idx];  y_val = y_tr[val_idx]
            val_pw = torch.tensor([float((y_val.cpu().numpy()==0).sum()/
                                         max((y_val.cpu().numpy()==1).sum(),1))],
                                  dtype=torch.float32).to(device)
            vl_fn  = nn.BCEWithLogitsLoss(pos_weight=val_pw)
            eff_bs2 = max(2, min(BATCH_SIZE, len(X_tr2)-1))
            dl2 = DataLoader(TensorDataset(X_tr2, y_tr2), batch_size=eff_bs2,
                             shuffle=True, drop_last=True)
            best_vl = float("inf"); pat_ct = 0; best_st = None
            model.train()
            for ep in range(EPOCHS):
                for xb, yb in dl2:
                    opt.zero_grad(); loss = loss_fn(model(xb), yb)
                    loss.backward(); opt.step()
                model.eval()
                with torch.no_grad(): vl = vl_fn(model(X_val), y_val).item()
                if vl < best_vl - MIN_DELTA:
                    best_vl = vl; pat_ct = 0; best_st = copy.deepcopy(model.state_dict())
                else:
                    pat_ct += 1
                    if pat_ct >= PATIENCE:
                        print(f"    Early stop ep {ep+1}/{EPOCHS} val_loss={best_vl:.4f}"); break
                model.train()
            if best_st is not None: model.load_state_dict(best_st)
            tl = time.time() - tt

            ti = time.time(); model.eval()
            with torch.no_grad():
                infer_bs = max(256, BATCH_SIZE)
                probs_list = []
                for i in range(0, len(X_te), infer_bs):
                    probs_list.append(torch.sigmoid(model(X_te[i:i+infer_bs])).cpu())
                probs = torch.cat(probs_list).numpy()
            il = time.time() - ti; preds = (probs >= 0.5).astype(float)

            cr = _proc.memory_info().rss
            if cr > peak: peak = cr
            fold_mem_mb = max(0.0, (cr - mem_fold_start) / 1024 / 1024)

            np_ = sum(p.numel() for p in model.parameters())
            fm = calculate_metrics(y_te_np, preds, probs)
            fm["train_latency_s"] = tl; fm["infer_latency_ms"] = il*1000/max(len(y_te_np),1)
            fm["total_latency_s"] = tl+il; fm["total_mem_mb"] = fold_mem_mb
            fm["n_params"] = np_; fm["ece"] = calculate_ece(y_te_np, probs)

            _pred_dir = os.path.join(RESULTS_DIR, "predictions")
            os.makedirs(_pred_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(_pred_dir, f"seed{seed}_fold{oi}.npz"),
                y_true=y_te_np, y_pred=preds, test_idx=tei)
            for k in sm:
                if k in fm: sm[k].append(fm[k])
            _save_partial(RESULTS_DIR, seed, oi, sm)

            if fm["f1"] > best_s:
                best_s = fm["f1"]
                torch.save({"model_state": model.state_dict()},
                           os.path.join(CHECKPOINT_DIR, f"transformer_all_seed{seed}_best.pt"))

            print(f"  >>> Fold {oi+1}({tl:.1f}s): Acc={fm['acc']:.4f} MCC={fm['mcc']:.4f} "
                  f"F1={fm['f1']:.4f} AUC={fm['auc']:.4f}")

            if oi == n_outer - 1:
                plot_confusion_matrix(y_te_np, preds, f"Transformer CM(Seed={seed})",
                    save_path=os.path.join(RESULTS_DIR, f"cm_transformer_all_seed{seed}.png"))
                plot_reliability_diagram(y_te_np, probs, f"Transformer Calib(Seed={seed})",
                    save_path=os.path.join(RESULTS_DIR,
                                           f"reliability_transformer_all_seed{seed}.png"))

        print(f"\n   Seed {seed} done in {time.time()-t_stage:.1f}s")
        f1s = sm.get("f1", [])
        if len(f1s) >= 3: _, ps = sp_stats.shapiro(f1s); shaps.append(ps)
        print_stage_report(STAGE_KEY, seed, sm, fold_shapiro_ps=shaps, mann_whitney_p=1.0)
        _clear_partial(RESULTS_DIR, seed)
        all_results[seed] = {STAGE_KEY: sm}; save_results_json(all_results, jp)

    print_final_aggregation_report(all_results, seeds, [STAGE_KEY])
    save_results_json(all_results, jp)
    tt = time.time()-t0; pm = (peak-mem0)/1024/1024
    print(f"\n{'='*60}\n  Latency:{int(tt//3600):02d}h{int((tt%3600)//60):02d}m{int(tt%60):02d}s"
          f"  Memory:{pm:.1f}MB\n  Results:{jp}\n{'='*60}\nTransformer(Full Data) done!")


if __name__ == "__main__":
    run_pipeline()
