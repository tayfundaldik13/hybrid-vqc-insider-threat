import os
import sys
import random
import numpy as np
import re
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src import config as cfg
from src.data_process import prepare_spedia_embeddings, create_tuning_eval_split
from src.models.reshybnet import ResHybnet, build_knn_adj

SEED       = 42
TUNING_N   = 100
N_CV_FOLDS = 3
N_TRIALS   = 90
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "config.py")
INPUT_DIM  = 16


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tuning_pool():
    X, y = prepare_spedia_embeddings(cfg.SPEDIA_FEATURES_PATH)
    tuning, _ = create_tuning_eval_split(
        X, y, tuning_size=TUNING_N, seed=SEED, threat_ratio=0.4
    )
    X_tune, y_tune = tuning[0], tuning[1]
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})")
    return X_tune, y_tune


def _update_config(key, value, quote_str=False):
    with open(CONFIG_PATH) as f:
        lines = f.readlines()
    updated = []
    found = False
    for line in lines:
        if re.match(rf"^\s*{re.escape(key)}\s*=" , line):
            updated.append(f"{key} = {value!r}\n" if quote_str else f"{key} = {value}\n")
            found = True
        else:
            updated.append(line)
    if not found:
        updated.append(f"{key} = {value!r}\n" if quote_str else f"{key} = {value}\n")
    with open(CONFIG_PATH, "w") as f:
        f.writelines(updated)


def _train_eval(X_tr, y_tr, X_vl, y_vl, params, device):
    set_all_seeds(SEED)
    A = build_knn_adj(X_tr, X_vl, k=params["k_neighbors"]).to(device)
    X_all = torch.tensor(np.vstack([X_tr, X_vl]), dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)

    model = ResHybnet(
        input_dim=INPUT_DIM,
        gcn_hidden=params["gcn_hidden"],
        gcn_layers=params["gcn_layers"],
        gcn_dropout=params["gcn_dropout"],
        cnn_channels=params["cnn_channels"],
        cnn_kernel=params["cnn_kernel"],
        mlp_hidden=params["mlp_hidden"],
        dropout=params["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    pos_w = torch.tensor([
        float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    ], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    n_tr = len(X_tr)
    model.train()
    for _ in range(params["epochs"]):
        optimizer.zero_grad()
        logits = model(A, X_all)
        loss = loss_fn(logits[:n_tr], y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(A, X_all)[n_tr:]).cpu().numpy().flatten()
    preds = (probs >= 0.5).astype(float)
    return f1_score(y_vl, preds, average="macro", zero_division=0)


def main():
    print("=" * 60)
    print("OPTUNA RESHYBNET TUNING - SPEDIA (50-500 pipeline)")
    print(f"   Tuning pool: {TUNING_N} samples (40% threat)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Objective: macro-F1")
    print(f"   Config target: {CONFIG_PATH}")
    print("=" * 60)

    set_all_seeds(SEED)
    X_tune, y_tune = get_tuning_pool()
    device = cfg.DEVICE

    def objective(trial):
        params = {
            "k_neighbors":    trial.suggest_int("k_neighbors", 2, 10),
            "gcn_hidden":     trial.suggest_categorical("gcn_hidden", [16, 32, 64, 128]),
            "gcn_layers":     trial.suggest_int("gcn_layers", 1, 3),
            "gcn_dropout":    trial.suggest_float("gcn_dropout", 0.1, 0.5),
            "cnn_channels":   trial.suggest_categorical("cnn_channels", [8, 16, 32]),
            "cnn_kernel":     trial.suggest_int("cnn_kernel", 2, 7),
            "mlp_hidden":     trial.suggest_categorical("mlp_hidden", [16, 32, 64]),
            "dropout":        trial.suggest_float("dropout", 0.1, 0.5),
            "lr":             trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay":   trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs":         trial.suggest_int("epochs", 20, 100),
        }

        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s = []
        for tr_idx, vl_idx in skf.split(X_tune, y_tune):
            f1 = _train_eval(X_tune[tr_idx], y_tune[tr_idx],
                             X_tune[vl_idx], y_tune[vl_idx], params, device)
            f1s.append(f1)
        return float(np.mean(f1s))

    study = optuna.create_study(
        direction="maximize",
        study_name="reshybnet_SPEDIA_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    p = study.best_params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={study.best_value:.4f}")
    print(f"\nUpdating {CONFIG_PATH}")
    print(f"{'-'*60}")
    _update_config("RESHYBNET_K_NEIGHBORS", p["k_neighbors"])
    _update_config("RESHYBNET_GCN_HIDDEN", p["gcn_hidden"])
    _update_config("RESHYBNET_GCN_LAYERS", p["gcn_layers"])
    _update_config("RESHYBNET_GCN_DROPOUT", p["gcn_dropout"])
    _update_config("RESHYBNET_CNN_CHANNELS", p["cnn_channels"])
    _update_config("RESHYBNET_CNN_KERNEL", p["cnn_kernel"])
    _update_config("RESHYBNET_MLP_HIDDEN", p["mlp_hidden"])
    _update_config("RESHYBNET_DROPOUT", p["dropout"])
    _update_config("RESHYBNET_LR", p["lr"])
    _update_config("RESHYBNET_EPOCHS", p["epochs"])
    _update_config("RESHYBNET_WEIGHT_DECAY", p["weight_decay"])
    print(f"   RESHYBNET_K_NEIGHBORS  = {p['k_neighbors']}")
    print(f"   RESHYBNET_GCN_HIDDEN   = {p['gcn_hidden']}")
    print(f"   RESHYBNET_GCN_LAYERS   = {p['gcn_layers']}")
    print(f"   RESHYBNET_GCN_DROPOUT  = {p['gcn_dropout']:.6f}")
    print(f"   RESHYBNET_CNN_CHANNELS = {p['cnn_channels']}")
    print(f"   RESHYBNET_CNN_KERNEL   = {p['cnn_kernel']}")
    print(f"   RESHYBNET_MLP_HIDDEN   = {p['mlp_hidden']}")
    print(f"   RESHYBNET_DROPOUT      = {p['dropout']:.6f}")
    print(f"   RESHYBNET_LR           = {p['lr']:.6f}")
    print(f"   RESHYBNET_EPOCHS       = {p['epochs']}")
    print(f"   RESHYBNET_WEIGHT_DECAY = {p['weight_decay']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
