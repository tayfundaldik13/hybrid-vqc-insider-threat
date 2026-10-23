import os
import sys
import random
import numpy as np
import re
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src import config as cfg
from src.data_process import prepare_spedia_embeddings, create_tuning_eval_split
from src.models.mlp import SimpleMLP

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
    model = SimpleMLP(
        input_dim=INPUT_DIM,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
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

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32).to(device)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32).to(device)

    eff_bs = max(2, min(params["batch_size"], len(X_tr_t) - 1))
    train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=eff_bs, shuffle=True, drop_last=True)

    model.train()
    for _ in range(params["epochs"]):
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_vl_t)).cpu().numpy().flatten()
    preds = (probs >= 0.5).astype(float)
    return f1_score(y_vl, preds, average="macro", zero_division=0)


def main():
    print("=" * 60)
    print("OPTUNA MLP TUNING - SPEDIA (50-500 pipeline)")
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
            "hidden_size":  trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 1, 4),
            "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
            "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs":       trial.suggest_int("epochs", 20, 100),
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
        study_name="mlp_SPEDIA_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    p = study.best_params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={study.best_value:.4f}")
    print(f"\nUpdating {CONFIG_PATH}")
    print(f"{'-'*60}")
    _update_config("MLP_HIDDEN_SIZE", p["hidden_size"])
    _update_config("MLP_NUM_LAYERS", p["num_layers"])
    _update_config("MLP_DROPOUT", p["dropout"])
    _update_config("MLP_LR", p["lr"])
    _update_config("MLP_BATCH_SIZE", p["batch_size"])
    _update_config("MLP_EPOCHS", p["epochs"])
    _update_config("MLP_WEIGHT_DECAY", p["weight_decay"])
    print(f"   MLP_HIDDEN_SIZE  = {p['hidden_size']}")
    print(f"   MLP_NUM_LAYERS   = {p['num_layers']}")
    print(f"   MLP_DROPOUT      = {p['dropout']:.6f}")
    print(f"   MLP_LR           = {p['lr']:.6f}")
    print(f"   MLP_BATCH_SIZE   = {p['batch_size']}")
    print(f"   MLP_EPOCHS       = {p['epochs']}")
    print(f"   MLP_WEIGHT_DECAY = {p['weight_decay']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
