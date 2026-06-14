"""
Quantum VQC Ablation Study for CERT r4.2
"""
import os
import time
import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats as sp_stats
import psutil, os as _os

from src import config
from src.models.quantum_vqc import HybridQuantumModel
from src.data_process import (prepare_cert_embeddings,
    create_nonoverlapping_stages, create_nested_folds)
from src.utils import (
    calculate_metrics, calculate_ece,
    print_stage_report, save_results_json
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_ablation")
BASELINE_JSON = os.path.join(BASE_DIR, "results_3", "quantum", "quantum_results.json")

LR         = 0.010438
EPOCHS     = 40
BATCH_SIZE = 5
THRESHOLD  = 0.5

BASELINE_CFG = dict(n_qubits=9, n_layers=2, nn_hidden=8, nn_depth=1, pre_hidden=0, pca_dim=16)

ABLATION_CONFIGS = {
    "qubits_4":      dict(n_qubits=4, n_layers=2, nn_hidden=8,  nn_depth=1, pre_hidden=0,  pca_dim=16),
    "qubits_6":      dict(n_qubits=6, n_layers=2, nn_hidden=8,  nn_depth=1, pre_hidden=0,  pca_dim=16),
    "qubits_9":      dict(n_qubits=9, n_layers=2, nn_hidden=8,  nn_depth=1, pre_hidden=0,  pca_dim=16),
    
    "layers_1":      dict(n_qubits=9, n_layers=1, nn_hidden=8,  nn_depth=1, pre_hidden=0,  pca_dim=16),
    "layers_3":      dict(n_qubits=9, n_layers=3, nn_hidden=8,  nn_depth=1, pre_hidden=0,  pca_dim=16),

    "post_none":     dict(n_qubits=9, n_layers=2, nn_hidden=8,  nn_depth=0, pre_hidden=0,  pca_dim=16),
    "post_wide":     dict(n_qubits=9, n_layers=2, nn_hidden=32, nn_depth=1, pre_hidden=0,  pca_dim=16),
    "post_deep":     dict(n_qubits=9, n_layers=2, nn_hidden=16, nn_depth=2, pre_hidden=0,  pca_dim=16),
    
    "pre_hidden_32": dict(n_qubits=9, n_layers=2, nn_hidden=8,  nn_depth=1, pre_hidden=32, pca_dim=16),
    "pre_hidden_64": dict(n_qubits=9, n_layers=2, nn_hidden=8,  nn_depth=1, pre_hidden=64, pca_dim=16),
    
    "pca_8":         dict(n_qubits=9, n_layers=2, nn_hidden=8,  nn_depth=1, pre_hidden=0,  pca_dim=8),
}

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_eval(cfg: dict, X_train_np, y_train_np, X_test_np, y_test_np, seed: int):
    pca_dim = cfg["pca_dim"]
    X_tr = torch.tensor(X_train_np[:, :pca_dim], dtype=torch.float32)
    X_te = torch.tensor(X_test_np[:, :pca_dim],  dtype=torch.float32)
    y_tr = torch.tensor(y_train_np, dtype=torch.float32)
    y_te = torch.tensor(y_test_np,  dtype=torch.float32)

    set_all_seeds(seed)
    model = HybridQuantumModel(
        input_dim=pca_dim,
        n_qubits=cfg["n_qubits"],
        n_layers=cfg["n_layers"],
        nn_hidden=cfg["nn_hidden"],
        nn_depth=cfg["nn_depth"],
        pre_hidden=cfg["pre_hidden"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.BCEWithLogitsLoss()

    eff_bs = max(2, min(BATCH_SIZE, len(X_tr) - 1))
    dl = DataLoader(TensorDataset(X_tr, y_tr),
                    batch_size=eff_bs, shuffle=True, drop_last=True)

    t0 = time.time()
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            logits = model(xb)
            loss   = loss_fn(logits, yb.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_latency = time.time() - t0

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        probs   = torch.sigmoid(model(X_te)).cpu().numpy().flatten()
        preds   = (probs >= THRESHOLD).astype(float)
        targets = y_te.cpu().numpy().flatten()
    infer_latency = time.time() - t1

    n_params = sum(p.numel() for p in model.parameters())
    fm = calculate_metrics(targets, preds, probs)
    fm["train_latency_s"]  = train_latency
    fm["infer_latency_ms"] = infer_latency * 1000 / max(len(targets), 1)
    fm["total_latency_s"]  = train_latency + infer_latency
    fm["total_mem_mb"]     = 0.0
    fm["n_params"]         = n_params
    fm["ece"]              = calculate_ece(targets, probs)
    return fm, targets, preds, probs


def run_ablation():
    _proc = psutil.Process(_os.getpid())
    pipeline_start = time.time()
    mem_start = _proc.memory_info().rss
    peak_mem  = mem_start

    data_sizes = config.DATA_SIZES
    seeds      = config.SEEDS
    n_outer    = config.N_OUTER_FOLDS
    total_pool = sum(data_sizes)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "quantum_ablation_results.json")

    if os.path.exists(json_path):
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"Loaded existing ablation results: {list(all_results.keys())}")
    else:
        all_results = {}

    if "baseline" not in all_results:
        if os.path.exists(BASELINE_JSON):
            with open(BASELINE_JSON) as f:
                raw = json.load(f)
            baseline_by_seed_size = {
                str(seed): {str(sz): v for sz, v in sv.items()}
                for seed, sv in raw.items()
            }
            all_results["baseline"] = baseline_by_seed_size
            print(f"Baseline loaded from {BASELINE_JSON}")
            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            print(f"WARNING: baseline JSON not found at {BASELINE_JSON}")

    for cfg_name, cfg in ABLATION_CONFIGS.items():
        print(f"\n{'#'*60}")
        print(f"Ablation Configuration: {cfg_name}")
        print(f"  {cfg}")
        print(f"{'#'*60}")

        if cfg_name not in all_results:
            all_results[cfg_name] = {}

        for seed_idx, seed in enumerate(seeds):
            seed_key = str(seed)
            if seed_key in all_results[cfg_name]:
                print(f"Seed {seed} already done — skipping.")
                continue

            print(f"\nSeed {seed} ({seed_idx+1}/{len(seeds)})")
            set_all_seeds(seed)

            full_embeddings, full_labels, full_texts = prepare_cert_embeddings(
                max_size=total_pool, seed=seed
            )
            stages = create_nonoverlapping_stages(
                full_embeddings, full_labels, data_sizes, seed=seed, texts=full_texts
            )

            all_results[cfg_name][seed_key] = {}

            for stage_idx, data_size in enumerate(data_sizes):
                stage_start = time.time()
                sz_key = str(data_size)
                print(f"\n  {'='*50}")
                print(f"  Size={data_size}, Seed={seed}, Config={cfg_name}")

                stage_emb, stage_labels, _ = stages[data_size]
                nested_folds = create_nested_folds(stage_labels, n_outer, n_inner=1, seed=seed)

                stage_metrics = {
                    "acc": [], "kappa": [], "mcc": [], "precision": [],
                    "recall": [], "specificity": [], "f1": [], "macro_f1": [], "auc": [],
                    "ece": [],
                    "train_latency_s": [], "infer_latency_ms": [],
                    "total_latency_s": [], "total_mem_mb": [], "n_params": []
                }

                for outer_idx, fold_data in enumerate(nested_folds):
                    outer_train_idx = fold_data["outer_train"]
                    outer_test_idx  = fold_data["outer_test"]

                    set_all_seeds(seed)
                    fm, targets, preds, probs = train_and_eval(
                        cfg,
                        stage_emb[outer_train_idx], stage_labels[outer_train_idx],
                        stage_emb[outer_test_idx],  stage_labels[outer_test_idx],
                        seed
                    )

                    current_rss = _proc.memory_info().rss
                    if current_rss > peak_mem:
                        peak_mem = current_rss

                    for key in stage_metrics:
                        if key in fm:
                            stage_metrics[key].append(fm[key])

                    print(f"Fold {outer_idx+1} (train {fm['train_latency_s']:.1f}s): "
                          f"Acc={fm['acc']:.4f}  MCC={fm['mcc']:.4f}  "
                          f"F1={fm['f1']:.4f}  AUC={fm['auc']:.4f}  ECE={fm['ece']:.4f}")

                stage_time = time.time() - stage_start
                print(f"  Stage {data_size} done in {stage_time:.1f}s")
                all_results[cfg_name][seed_key][sz_key] = stage_metrics

            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved: {json_path}")

    print(f"\n{'='*70}")
    print("Ablation Summary (mean across seeds × folds)")
    print(f"{'='*70}")

    all_cfg_names = ["baseline"] + list(ABLATION_CONFIGS.keys())
    for sz in data_sizes:
        print(f"\n--- Size={sz} ---")
        header = f"{'Config':<16} {'F1':>7} {'Recall':>8} {'Prec':>7} {'Spec':>7} {'AUC':>7} {'MCC':>7} {'ECE':>7} {'Params':>8}"
        print(header)
        print('-' * 80)
        for cfg_name in all_cfg_names:
            if cfg_name not in all_results:
                continue
            d = all_results[cfg_name]
            row = {}
            for m in ["f1","recall","precision","specificity","auc","mcc","ece","n_params"]:
                vals = []
                for seed_str, sizes_dict in d.items():
                    sz_str = str(sz)
                    if sz_str in sizes_dict and m in sizes_dict[sz_str]:
                        vals.extend(sizes_dict[sz_str][m])
                row[m] = np.mean(vals) if vals else float("nan")
            print(f"{cfg_name:<16} {row['f1']:>7.4f} {row['recall']:>8.4f} "
                  f"{row['precision']:>7.4f} {row['specificity']:>7.4f} "
                  f"{row['auc']:>7.4f} {row['mcc']:>7.4f} "
                  f"{row['ece']:>7.4f} {row['n_params']:>8.0f}")

    total_time  = time.time() - pipeline_start
    peak_mem_mb = (peak_mem - mem_start) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f" Total Ablation Time: {int(total_time//3600):02d}h "
          f"{int((total_time%3600)//60):02d}m {int(total_time%60):02d}s")
    print(f"Peak Memory: {peak_mem_mb:.1f} MB")
    print(f"Results: {json_path}")
    print("Ablation study complete!")


if __name__ == "__main__":
    run_ablation()
