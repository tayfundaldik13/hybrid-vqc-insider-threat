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

from src import config
from src.data_process import load_behavioral_features, create_tuning_eval_split
from src.data_process import GroupAwareStratifiedKFold as StratifiedKFold, use_full_pool, multi_pool_folds
use_full_pool()
from src.models.transformer import BehaviorTransformer

SEED       = 42
TUNING_SEEDS = int(os.environ.get('TUNING_SEEDS', '3'))
POOLS = []
TUNING_N   = 100
N_CV_FOLDS = 3
N_TRIALS   = 30
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "config.py")
INPUT_DIM  = 16

VALID_HEAD_CONFIGS = [
    (16, 2), (16, 4), (16, 8),
    (32, 2), (32, 4), (32, 8),
    (64, 2), (64, 4), (64, 8),
    (128, 2), (128, 4), (128, 8),
]


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tuning_pool():
    X, y = load_behavioral_features()
    tuning, _ = create_tuning_eval_split(
        X, y, tuning_size=TUNING_N, seed=SEED, threat_ratio=0.4
    )
    X_tune, y_tune = tuning[0], tuning[1]
    POOLS[:] = [(X_tune, y_tune)] + [
        create_tuning_eval_split(X, y, tuning_size=TUNING_N, seed=SEED + _i, threat_ratio=0.4)[0][:2]
        for _i in range(1, TUNING_SEEDS)]
    print(f"   Tuning pools ({len(POOLS)}): threats per pool = {[int(p[1].sum()) for p in POOLS]}, size = {[len(p[1]) for p in POOLS]}")
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
    d_model, n_heads = params["d_model"], params["n_heads"]
    model = BehaviorTransformer(
        input_dim=INPUT_DIM,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        dim_feedforward=params["dim_feedforward"],
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
    print("OPTUNA TRANSFORMER TUNING - r4.2 (50-500 pipeline)")
    print(f"   Tuning pool: {TUNING_N} samples (20% threat / 80% normal)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Objective: macro-F1")
    print(f"   Config target: {CONFIG_PATH}")
    print("=" * 60)

    set_all_seeds(SEED)
    X_tune, y_tune = get_tuning_pool()
    device = config.DEVICE

    def objective(trial):
        cfg = trial.suggest_categorical("d_model_n_heads", [f"{d}x{h}" for d, h in VALID_HEAD_CONFIGS])
        d_model, n_heads = map(int, cfg.split("x"))
        params = {
            "d_model":        d_model,
            "n_heads":        n_heads,
            "n_layers":       trial.suggest_int("n_layers", 1, 4),
            "dropout":        trial.suggest_float("dropout", 0.1, 0.5),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [16, 32, 64, 128]),
            "lr":             trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size":     trial.suggest_categorical("batch_size", [8, 16, 32]),
            "weight_decay":   trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs":         trial.suggest_int("epochs", 20, 100),
        }

        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s = []
        for X_tune, y_tune, tr_idx, vl_idx in multi_pool_folds(POOLS, N_CV_FOLDS, SEED, StratifiedKFold):
            f1 = _train_eval(X_tune[tr_idx], y_tune[tr_idx],
                             X_tune[vl_idx], y_tune[vl_idx], params, device)
            f1s.append(f1)
        trial.set_user_attr("f1_std", float(np.std(f1s)))
        return float(np.mean(f1s))

    study = optuna.create_study(
        direction="maximize",
        study_name="transformer_r4.2_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    p = study.best_params
    d_model, n_heads = map(int, p["d_model_n_heads"].split("x"))
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={study.best_value:.4f} (+/-{study.best_trial.user_attrs.get('f1_std', 0):.4f})")
    print(f"\nUpdating {CONFIG_PATH}")
    print(f"{'-'*60}")
    _update_config("TF_D_MODEL", d_model)
    _update_config("TF_N_HEADS", n_heads)
    _update_config("TF_N_LAYERS", p["n_layers"])
    _update_config("TF_DROPOUT", p["dropout"])
    _update_config("TF_DIM_FEEDFORWARD", p["dim_feedforward"])
    _update_config("TF_LR", p["lr"])
    _update_config("TF_BATCH_SIZE", p["batch_size"])
    _update_config("TF_EPOCHS", p["epochs"])
    _update_config("TF_WEIGHT_DECAY", p["weight_decay"])
    print(f"   TF_D_MODEL         = {d_model}")
    print(f"   TF_N_HEADS         = {n_heads}")
    print(f"   TF_N_LAYERS        = {p['n_layers']}")
    print(f"   TF_DROPOUT         = {p['dropout']:.6f}")
    print(f"   TF_DIM_FEEDFORWARD = {p['dim_feedforward']}")
    print(f"   TF_LR              = {p['lr']:.6f}")
    print(f"   TF_BATCH_SIZE      = {p['batch_size']}")
    print(f"   TF_EPOCHS          = {p['epochs']}")
    print(f"   TF_WEIGHT_DECAY    = {p['weight_decay']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
