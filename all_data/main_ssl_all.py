import os, sys, time, random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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

RESULTS_DIR    = os.path.join(_ALL_DATA_DIR, "results_all", "ssl")
CHECKPOINT_DIR = os.path.join(_ALL_DATA_DIR, "checkpoints_all", "ssl")
INPUT_DIM      = 16

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
STAGE_KEY        = "all"
PATIENCE         = config.EARLY_STOP_PATIENCE
MIN_DELTA        = config.EARLY_STOP_MIN_DELTA
VAL_FRAC         = config.EARLY_STOP_VAL_FRAC


def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def corrupt_features(X, corruption_rate, marginals):
    mask = torch.rand_like(X) < corruption_rate
    rand_idx = torch.randint(0, len(marginals), (X.shape[0],))
    X_corrupt = marginals[rand_idx].to(X.device)
    return torch.where(mask, X_corrupt, X)


def nt_xent_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    n = z1.shape[0]; z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2*n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    labels = torch.cat([torch.arange(n, device=z.device)+n,
                         torch.arange(n, device=z.device)], dim=0)
    return F.cross_entropy(sim, labels)


def nt_xent_loss_batched(model, X, marginals, corruption_rate, temperature, batch_size=1024):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            xb_c = corrupt_features(xb, corruption_rate, marginals).to(X.device)
            losses.append(nt_xent_loss(model.project(xb), model.project(xb_c), temperature).item())
    return float(np.mean(losses)) if losses else float("inf")


class SCARFEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, n_layers, dropout):
        super().__init__()
        layers = []; in_d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]; in_d = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.projector = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.ReLU(),
                                        nn.Linear(proj_dim, proj_dim))
    def encode(self, x): return self.encoder(x)
    def project(self, x): return self.projector(self.encode(x))


class SSLClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder; self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x): return self.head(self.encoder.encode(x)).squeeze(-1)


def run_pipeline():
    import psutil, os as _os
    _proc = psutil.Process(_os.getpid()); t0 = time.time()
    mem0 = _proc.memory_info().rss; peak = mem0
    device = config.DEVICE; seeds = config.SEEDS; n_outer = config.N_OUTER_FOLDS

    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Self-Supervised (SCARF) PIPELINE - FULL DATASET")
    print(f"   Encoder: {N_ENCODER_LAYERS}xLinear({HIDDEN_DIM})+BN+ReLU  "
          f"Proj:{HIDDEN_DIM}->{PROJ_DIM}")
    print(f"   Corruption:{CORRUPTION_RATE:.2f}  Temperature:{TEMPERATURE:.2f}")
    print(f"   Pretrain: lr={PRETRAIN_LR:.2e} ep={PRETRAIN_EPOCHS}  "
          f"Finetune: lr={FINETUNE_LR:.2e} ep={FINETUNE_EPOCHS}  "
          f"FreezeEnc:{FREEZE_ENCODER}")
    print(f"   Seeds:{seeds}  |  Nested CV:{n_outer}  |  Device:{device}\n{'='*60}\n")

    full_emb, full_lab = load_behavioral_features()
    print(f"   Full dataset:{len(full_lab)} samples "
          f"(threat={int(full_lab.sum())},ratio={full_lab.mean():.4f})\n")

    jp = os.path.join(RESULTS_DIR, "ssl_all_results.json")
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

            X_tr = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
            X_te = torch.tensor(X_te_np, dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
            marginals = X_tr.cpu()
            pw = torch.tensor([float((y_tr_np==0).sum()/max((y_tr_np==1).sum(),1))],
                               dtype=torch.float32).to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

            eff_bs = max(4, min(BATCH_SIZE, len(X_tr)-1))

            set_all_seeds(seed)
            encoder = SCARFEncoder(INPUT_DIM, HIDDEN_DIM, PROJ_DIM,
                                    N_ENCODER_LAYERS, DROPOUT).to(device)
            opt_pre = torch.optim.Adam(encoder.parameters(), lr=PRETRAIN_LR)
            dl_pre  = DataLoader(TensorDataset(X_tr), batch_size=eff_bs,
                                  shuffle=True, drop_last=True)

            import copy
            n_pre_val = max(2, int(len(X_tr) * VAL_FRAC))
            pre_val_idx = np.random.choice(len(X_tr), n_pre_val, replace=False)
            pre_tr_idx  = np.setdiff1d(np.arange(len(X_tr)), pre_val_idx)
            X_ptv = X_tr[pre_val_idx]; X_ptr = X_tr[pre_tr_idx]
            dl_pre = DataLoader(TensorDataset(X_ptr), batch_size=eff_bs,
                                shuffle=True, drop_last=True)
            tt = time.time(); encoder.train()
            best_pre=float("inf"); pat_pre=0; best_enc=None
            for ep in range(PRETRAIN_EPOCHS):
                for (xb,) in dl_pre:
                    x_c = corrupt_features(xb, CORRUPTION_RATE, marginals).to(device)
                    loss = nt_xent_loss(encoder.project(xb), encoder.project(x_c), TEMPERATURE)
                    opt_pre.zero_grad(); loss.backward(); opt_pre.step()
                encoder.eval()
                vl_pre = nt_xent_loss_batched(
                    encoder, X_ptv, marginals, CORRUPTION_RATE, TEMPERATURE, batch_size=1024
                )
                if vl_pre < best_pre - MIN_DELTA:
                    best_pre=vl_pre; pat_pre=0; best_enc=copy.deepcopy(encoder.state_dict())
                else:
                    pat_pre+=1
                    if pat_pre>=PATIENCE:
                        print(f"    Pretrain early stop ep {ep+1}/{PRETRAIN_EPOCHS}"); break
                encoder.train()
            if best_enc is not None: encoder.load_state_dict(best_enc)

            if FREEZE_ENCODER:
                for p in encoder.encoder.parameters(): p.requires_grad = False
            clf = SSLClassifier(encoder, HIDDEN_DIM).to(device)
            opt_ft = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, clf.parameters()), lr=FINETUNE_LR)
            pos_i=np.where(y_tr_np==1)[0]; neg_i=np.where(y_tr_np==0)[0]
            n_ftv=max(2,int(len(y_tr_np)*VAL_FRAC))
            n_vp=max(1,int(n_ftv*y_tr_np.mean())); n_vn=n_ftv-n_vp
            rng=np.random.default_rng(seed)
            ft_val_idx=np.concatenate([rng.choice(pos_i,min(n_vp,len(pos_i)),replace=False),
                                        rng.choice(neg_i,min(n_vn,len(neg_i)),replace=False)])
            ft_tr_idx=np.setdiff1d(np.arange(len(y_tr_np)),ft_val_idx)
            X_ftv=X_tr[ft_val_idx]; y_ftv=y_tr[ft_val_idx]
            X_ftt=X_tr[ft_tr_idx];  y_ftt=y_tr[ft_tr_idx]
            val_pw=torch.tensor([float((y_ftv.cpu().numpy()==0).sum()/
                                        max((y_ftv.cpu().numpy()==1).sum(),1))],
                                dtype=torch.float32).to(device)
            vl_fn=nn.BCEWithLogitsLoss(pos_weight=val_pw)
            eff_ft=max(2,min(BATCH_SIZE,len(X_ftt)-1))
            dl_ft=DataLoader(TensorDataset(X_ftt,y_ftt),batch_size=eff_ft,shuffle=True,drop_last=True)
            clf.train()
            best_ft=float("inf"); pat_ft=0; best_clf=None
            for ep in range(FINETUNE_EPOCHS):
                for xb,yb in dl_ft:
                    opt_ft.zero_grad()
                    loss=loss_fn(clf(xb),yb)
                    loss.backward(); opt_ft.step()
                clf.eval()
                with torch.no_grad(): vl_ft=vl_fn(clf(X_ftv),y_ftv).item()
                if vl_ft<best_ft-MIN_DELTA:
                    best_ft=vl_ft; pat_ft=0; best_clf=copy.deepcopy(clf.state_dict())
                else:
                    pat_ft+=1
                    if pat_ft>=PATIENCE:
                        print(f"    Finetune early stop ep {ep+1}/{FINETUNE_EPOCHS}"); break
                clf.train()
            if best_clf is not None: clf.load_state_dict(best_clf)
            tl = time.time() - tt

            ti = time.time(); clf.eval()
            with torch.no_grad():
                infer_bs = max(256, BATCH_SIZE)
                probs_list = []
                for i in range(0, len(X_te), infer_bs):
                    probs_list.append(torch.sigmoid(clf(X_te[i:i+infer_bs])).cpu())
                probs = torch.cat(probs_list).numpy()
            il = time.time() - ti; preds = (probs >= 0.5).astype(float)

            cr = _proc.memory_info().rss
            if cr > peak: peak = cr
            fold_mem_mb = max(0.0, (cr - mem_fold_start) / 1024 / 1024)

            np_ = sum(p.numel() for p in clf.parameters())
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
                torch.save({"model_state": clf.state_dict()},
                           os.path.join(CHECKPOINT_DIR, f"ssl_all_seed{seed}_best.pt"))

            print(f"  >>> Fold {oi+1}({tl:.1f}s): Acc={fm['acc']:.4f} MCC={fm['mcc']:.4f} "
                  f"F1={fm['f1']:.4f} AUC={fm['auc']:.4f}")

            if oi == n_outer - 1:
                plot_confusion_matrix(y_te_np, preds, f"SSL-SCARF CM(Seed={seed})",
                    save_path=os.path.join(RESULTS_DIR, f"cm_ssl_all_seed{seed}.png"))
                plot_reliability_diagram(y_te_np, probs, f"SSL-SCARF Calib(Seed={seed})",
                    save_path=os.path.join(RESULTS_DIR, f"reliability_ssl_all_seed{seed}.png"))

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
          f"  Memory:{pm:.1f}MB\n  Results:{jp}\n{'='*60}\nSSL-SCARF(Full Data) done!")


if __name__ == "__main__":
    run_pipeline()
