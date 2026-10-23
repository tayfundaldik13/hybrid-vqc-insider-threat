import os, sys, time, random
import numpy as np
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
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

RESULTS_DIR    = os.path.join(_ALL_DATA_DIR, "results_all", "reshybnet")
CHECKPOINT_DIR = os.path.join(_ALL_DATA_DIR, "checkpoints_all", "reshybnet")
INPUT_DIM      = 16

K_NEIGHBORS  = config.RESHYBNET_K_NEIGHBORS
GCN_HIDDEN   = config.RESHYBNET_GCN_HIDDEN
GCN_LAYERS   = config.RESHYBNET_GCN_LAYERS
GCN_DROPOUT  = config.RESHYBNET_GCN_DROPOUT
CNN_CHANNELS = config.RESHYBNET_CNN_CHANNELS
CNN_KERNEL   = config.RESHYBNET_CNN_KERNEL
MLP_HIDDEN   = config.RESHYBNET_MLP_HIDDEN
DROPOUT      = config.RESHYBNET_DROPOUT
LR           = config.RESHYBNET_LR
EPOCHS       = config.RESHYBNET_EPOCHS
WEIGHT_DECAY = config.RESHYBNET_WEIGHT_DECAY
STAGE_KEY    = "all"
PATIENCE     = config.EARLY_STOP_PATIENCE
MIN_DELTA    = config.EARLY_STOP_MIN_DELTA
VAL_FRAC     = config.EARLY_STOP_VAL_FRAC

def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True

def build_knn_adj(X_tr, X_te, k):
    n_tr=len(X_tr); n_te=len(X_te); n=n_tr+n_te
    X_all=np.vstack([X_tr,X_te])
    nrm=np.linalg.norm(X_all,axis=1,keepdims=True)+1e-10; X_n=X_all/nrm
    nn_m=NearestNeighbors(n_neighbors=min(k+1,n_tr),metric="cosine",algorithm="auto",n_jobs=-1)
    nn_m.fit(X_n[:n_tr])
    r,c=[],[]
    _,idx=nn_m.kneighbors(X_n[:n_tr])
    for i,nb in enumerate(idx):
        for j in nb:
            if j!=i: r+=[i,j]; c+=[j,i]
    _,idx=nn_m.kneighbors(X_n[n_tr:])
    for it,nb in enumerate(idx):
        ig=n_tr+it
        for j in nb: r+=[ig,j]; c+=[j,ig]
    for i in range(n): r.append(i); c.append(i)
    A=sp.csr_matrix((np.ones(len(r),np.float32),(r,c)),shape=(n,n))
    d=np.asarray(A.sum(1)).flatten(); di=np.where(d>0,1/np.sqrt(d),0.0)
    D=sp.diags(di); A_n=(D@A@D).tocoo()
    idx=torch.tensor(np.vstack([A_n.row,A_n.col]),dtype=torch.long)
    val=torch.tensor(A_n.data,dtype=torch.float32)
    return torch.sparse_coo_tensor(idx,val,(n,n))

class GCNLayer(nn.Module):
    def __init__(self,i,o): super().__init__(); self.lin=nn.Linear(i,o)
    def forward(self,A,H): return self.lin(torch.sparse.mm(A,H))

class ResHybnet(nn.Module):
    def __init__(self,input_dim,gcn_hidden,gcn_layers,gcn_dropout,
                 cnn_channels,cnn_kernel,mlp_hidden,dropout):
        super().__init__()
        self.gcn_layers=nn.ModuleList(); self.gcn_drop=nn.Dropout(gcn_dropout)
        in_d=input_dim
        for _ in range(gcn_layers):
            self.gcn_layers.append(GCNLayer(in_d,gcn_hidden)); in_d=gcn_hidden
        ek=min(cnn_kernel,input_dim)
        self.cnn=nn.Sequential(nn.Conv1d(1,cnn_channels,kernel_size=ek,padding=ek//2),
                                nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        fused=gcn_hidden+cnn_channels
        self.fc1=nn.Linear(fused,mlp_hidden); self.fc_res=nn.Linear(fused,mlp_hidden)
        self.bn=nn.LayerNorm(mlp_hidden); self.drop=nn.Dropout(dropout)
        self.out=nn.Linear(mlp_hidden,1)
    def forward(self,A,X):
        h=X
        for l in self.gcn_layers: h=torch.relu(l(A,h)); h=self.gcn_drop(h)
        cnn_e=self.cnn(X.unsqueeze(1)).squeeze(-1)
        f=torch.cat([h,cnn_e],dim=-1)
        o=torch.relu(self.bn(self.fc1(f)+self.fc_res(f)))
        return self.out(self.drop(o)).squeeze(-1)

def run_pipeline():
    import psutil, os as _os
    _proc=psutil.Process(_os.getpid()); t0=time.time()
    mem0=_proc.memory_info().rss; peak=mem0
    device=config.DEVICE; seeds=config.SEEDS; n_outer=config.N_OUTER_FOLDS
    os.makedirs(RESULTS_DIR,exist_ok=True); os.makedirs(CHECKPOINT_DIR,exist_ok=True)
    print("ResHybnet PIPELINE - FULL DATASET (GCN + CNN)")
    print(f"   k={K_NEIGHBORS} gcn_h={GCN_HIDDEN} gcn_L={GCN_LAYERS} "
          f"cnn_ch={CNN_CHANNELS} mlp_h={MLP_HIDDEN} lr={LR:.5f} ep={EPOCHS}")
    print(f"   Seeds:{seeds}  |  Nested CV:{n_outer} folds  |  Device:{device}\n{'='*60}\n")
    full_emb,full_lab=load_behavioral_features()
    print(f"   Full dataset:{len(full_lab)} samples "
          f"(threat={int(full_lab.sum())},normal={int((full_lab==0).sum())},"
          f"ratio={full_lab.mean():.4f})\n")
    jp=os.path.join(RESULTS_DIR,"reshybnet_all_results.json")
    if os.path.exists(jp):
        import json as _j
        with open(jp) as _f: all_results={int(k):v for k,v in _j.load(_f).items()}
        print(f"Loaded seeds:{sorted(all_results.keys())}")
    else: all_results={}
    for si,seed in enumerate(seeds):
        if seed in all_results: print(f"\nSEED {seed} done-skip"); continue
        print(f"\n{'#'*60}\nSEED {seed} ({si+1}/{len(seeds)})\n{'#'*60}")
        set_all_seeds(seed)
        folds=create_nested_folds(full_lab,n_outer,n_inner=1,seed=seed)
        sm={"acc":[],"kappa":[],"mcc":[],"precision":[],"recall":[],"specificity":[],
            "f1":[],"macro_f1":[],"auc":[],"ece":[],"train_latency_s":[],
            "infer_latency_ms":[],"total_latency_s":[],"total_mem_mb":[],"n_params":[]}
        shaps=[]; best_s=-1.0; t_stage=time.time()

        _partial = _load_partial(RESULTS_DIR, seed)
        _start_fold = 0
        if _partial is not None:
            _start_fold = _partial["next_fold"]
            _pm = _partial["stage_metrics"]
            for _k in sm:
                if _k in _pm:
                    sm[_k] = list(_pm[_k])
            print(f"  [RESUME] Seed {seed}: resuming from fold {_start_fold+1}")
        for oi,fd in enumerate(folds):
            tri=fd["outer_train"]; tei=fd["outer_test"]
            print(f"\n  Outer Fold {oi+1}/{n_outer}: Train={len(tri)}, Test={len(tei)}")
            print(f"  Building graph...",end=" ",flush=True)
            mem_fold_start=_proc.memory_info().rss
            X_tr=full_emb[tri]; X_te=full_emb[tei]
            y_tr=full_lab[tri]; y_te=full_lab[tei]; n_tr=len(X_tr)
            tg=time.time(); A=build_knn_adj(X_tr,X_te,K_NEIGHBORS).to(device)
            print(f"done({time.time()-tg:.1f}s)")
            X_all=torch.tensor(np.vstack([X_tr,X_te]),dtype=torch.float32).to(device)
            y_tr_t=torch.tensor(y_tr,dtype=torch.float32).to(device)
            pw=torch.tensor([float((y_tr==0).sum()/max((y_tr==1).sum(),1))],
                            dtype=torch.float32).to(device)
            loss_fn=nn.BCEWithLogitsLoss(pos_weight=pw)
            set_all_seeds(seed)
            model=ResHybnet(INPUT_DIM,GCN_HIDDEN,GCN_LAYERS,GCN_DROPOUT,
                            CNN_CHANNELS,CNN_KERNEL,MLP_HIDDEN,DROPOUT).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
            import copy
            pos_tr_i = np.where(y_tr==1)[0]; neg_tr_i = np.where(y_tr==0)[0]
            n_val  = max(2, int(len(y_tr)*VAL_FRAC))
            n_vp   = max(1, int(n_val*y_tr.mean())); n_vn = n_val-n_vp
            rng    = np.random.default_rng(seed)
            val_idx2 = np.concatenate([rng.choice(pos_tr_i,min(n_vp,len(pos_tr_i)),replace=False),
                                       rng.choice(neg_tr_i,min(n_vn,len(neg_tr_i)),replace=False)])
            tr_idx2  = np.setdiff1d(np.arange(len(y_tr)), val_idx2)
            y_val_t  = y_tr_t[val_idx2]
            val_pw   = torch.tensor([float((y_val_t.cpu().numpy()==0).sum()/
                                           max((y_val_t.cpu().numpy()==1).sum(),1))],
                                    dtype=torch.float32).to(device)
            vl_fn    = nn.BCEWithLogitsLoss(pos_weight=val_pw)
            tt=time.time()
            best_vl=float("inf"); pat_ct=0; best_st=None
            model.train()
            for ep in range(EPOCHS):
                opt.zero_grad()
                loss=loss_fn(model(A,X_all)[tr_idx2], y_tr_t[tr_idx2])
                loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    vl=vl_fn(model(A,X_all)[val_idx2], y_val_t).item()
                if vl<best_vl-MIN_DELTA:
                    best_vl=vl; pat_ct=0; best_st=copy.deepcopy(model.state_dict())
                else:
                    pat_ct+=1
                    if pat_ct>=PATIENCE:
                        print(f"    Early stop ep {ep+1}/{EPOCHS} val_loss={best_vl:.4f}"); break
                model.train()
            if best_st is not None: model.load_state_dict(best_st)
            tl=time.time()-tt
            ti=time.time(); model.eval()
            with torch.no_grad(): probs=torch.sigmoid(model(A,X_all)[n_tr:]).cpu().numpy()
            il=time.time()-ti; preds=(probs>=0.5).astype(float)
            cr=_proc.memory_info().rss
            if cr>peak: peak=cr
            fold_mem_mb=max(0.0, (cr - mem_fold_start) / 1024 / 1024)
            np_=sum(p.numel() for p in model.parameters())
            fm=calculate_metrics(y_te,preds,probs)
            fm["train_latency_s"]=tl; fm["infer_latency_ms"]=il*1000/max(len(y_te),1)
            fm["total_latency_s"]=tl+il; fm["total_mem_mb"]=fold_mem_mb
            fm["n_params"]=np_; fm["ece"]=calculate_ece(y_te,probs)

            _pred_dir = os.path.join(RESULTS_DIR, "predictions")
            os.makedirs(_pred_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(_pred_dir, f"seed{seed}_fold{oi}.npz"),
                y_true=y_te, y_pred=preds, test_idx=tei)
            for k in sm:
                if k in fm: sm[k].append(fm[k])
            _save_partial(RESULTS_DIR, seed, oi, sm)
            if fm["f1"]>best_s:
                best_s=fm["f1"]
                torch.save({"model_state":model.state_dict()},
                           os.path.join(CHECKPOINT_DIR,f"reshybnet_all_seed{seed}_best.pt"))
            print(f"  >>> Fold {oi+1}({tl:.1f}s): Acc={fm['acc']:.4f} MCC={fm['mcc']:.4f} "
                  f"F1={fm['f1']:.4f} AUC={fm['auc']:.4f}")
            if oi==n_outer-1:
                plot_confusion_matrix(y_te,preds,f"ResHybnet CM(Seed={seed})",
                    save_path=os.path.join(RESULTS_DIR,f"cm_reshybnet_all_seed{seed}.png"))
                plot_reliability_diagram(y_te,probs,f"ResHybnet Calib(Seed={seed})",
                    save_path=os.path.join(RESULTS_DIR,f"reliability_reshybnet_all_seed{seed}.png"))
        print(f"\n   Seed {seed} done in {time.time()-t_stage:.1f}s")
        f1s=sm.get("f1",[])
        if len(f1s)>=3: _,ps=sp_stats.shapiro(f1s); shaps.append(ps)
        print_stage_report(STAGE_KEY,seed,sm,fold_shapiro_ps=shaps,mann_whitney_p=1.0)
        _clear_partial(RESULTS_DIR, seed)
        all_results[seed]={STAGE_KEY:sm}; save_results_json(all_results,jp)
    print_final_aggregation_report(all_results,seeds,[STAGE_KEY]); save_results_json(all_results,jp)
    tt=time.time()-t0; pm=(peak-mem0)/1024/1024
    print(f"\n{'='*60}\n  Latency:{int(tt//3600):02d}h{int((tt%3600)//60):02d}m{int(tt%60):02d}s"
          f"  Memory:{pm:.1f}MB\n  Results:{jp}\n{'='*60}\nResHybnet(Full Data) done!")

if __name__ == "__main__": run_pipeline()
