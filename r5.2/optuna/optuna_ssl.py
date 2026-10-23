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
from src.models.ssl import SCARFEncoder, SSLClassifier, corrupt_features, nt_xent_loss

SEED       = 42
TUNING_SEEDS = int(os.environ.get('TUNING_SEEDS', '3'))
POOLS = []
TUNING_N   = 100
N_CV_FOLDS = 3
N_TRIALS   = 30
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "config.py")
INPUT_DIM  = 16


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
    pos_w = float((y_tune == 0).sum() / max((y_tune == 1).sum(), 1))
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})  pos_weight={pos_w:.2f}")
    return X_tune, y_tune, pos_w


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


def main():
    print("=" * 60)
    print("OPTUNA TUNING - r5.2 (50-500 pipeline)")
    print(f"   Tuning pool: {TUNING_N} samples (20% threat / 80% normal)")
    print(f"   Phase 1: SCARF contrastive pretraining")
    print(f"   Phase 2: Supervised fine-tuning")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Config target: {CONFIG_PATH}")
    print("=" * 60)

    X_tune, y_tune, pos_weight_val = get_tuning_pool()
    device = config.DEVICE

    def objective(trial):
        hidden_dim       = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        proj_dim         = trial.suggest_categorical("proj_dim", [16, 32])
        n_encoder_layers = trial.suggest_int("n_encoder_layers", 2, 4)
        dropout          = trial.suggest_float("dropout", 0.1, 0.4)
        corruption_rate  = trial.suggest_float("corruption_rate", 0.1, 0.6)
        temperature      = trial.suggest_float("temperature", 0.1, 1.0)
        pretrain_lr      = trial.suggest_float("pretrain_lr", 1e-4, 1e-2, log=True)
        pretrain_epochs  = trial.suggest_categorical("pretrain_epochs", [50, 100])
        finetune_lr      = trial.suggest_float("finetune_lr", 1e-4, 1e-2, log=True)
        finetune_epochs  = trial.suggest_categorical("finetune_epochs", [30, 50, 100])
        batch_size       = trial.suggest_categorical("batch_size", [16, 32, 64])
        weight_decay     = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        freeze_encoder   = trial.suggest_categorical("freeze_encoder", [True, False])

        pos_w   = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        skf     = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s     = []

        for X_tune, y_tune, tr_idx, vl_idx in multi_pool_folds(POOLS, N_CV_FOLDS, SEED, StratifiedKFold):
            set_all_seeds(SEED)
            X_tr_np = X_tune[tr_idx]; X_vl_np = X_tune[vl_idx]
            y_tr_np = y_tune[tr_idx]; y_vl_np = y_tune[vl_idx]

            X_tr = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
            X_vl = torch.tensor(X_vl_np, dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
            marginals = X_tr.cpu()

            encoder = SCARFEncoder(INPUT_DIM, hidden_dim, proj_dim,
                                   n_encoder_layers, dropout).to(device)
            opt_pre = torch.optim.Adam(encoder.parameters(), lr=pretrain_lr)
            eff_bs  = max(4, min(batch_size, len(X_tr) - 1))
            dl_pre  = DataLoader(TensorDataset(X_tr),
                                 batch_size=eff_bs, shuffle=True, drop_last=True)

            encoder.train()
            for _ in range(pretrain_epochs):
                for (xb,) in dl_pre:
                    x_corrupt = corrupt_features(xb, corruption_rate, marginals).to(device)
                    z1 = encoder.project(xb)
                    z2 = encoder.project(x_corrupt)
                    loss = nt_xent_loss(z1, z2, temperature)
                    opt_pre.zero_grad(); loss.backward(); opt_pre.step()

            if freeze_encoder:
                for p in encoder.parameters():
                    p.requires_grad = False

            clf    = SSLClassifier(encoder, hidden_dim).to(device)
            opt_ft = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, clf.parameters()),
                lr=finetune_lr,
                weight_decay=weight_decay,
            )
            dl_ft  = DataLoader(TensorDataset(X_tr, y_tr),
                                batch_size=eff_bs, shuffle=True, drop_last=True)

            clf.train()
            for _ in range(finetune_epochs):
                for xb, yb in dl_ft:
                    opt_ft.zero_grad()
                    loss = loss_fn(clf(xb), yb)
                    loss.backward(); opt_ft.step()

            clf.eval()
            with torch.no_grad():
                probs = torch.sigmoid(clf(X_vl)).cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            f1s.append(f1_score(y_vl_np, preds, average="macro", zero_division=0))

        trial.set_user_attr("f1_std", float(np.std(f1s)))
        return float(np.mean(f1s))

    study = optuna.create_study(
        direction="maximize",
        study_name="ssl_r5.2_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial; p = best.params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"\nUpdating {CONFIG_PATH}")
    print(f"{'-'*60}")
    _update_config("SSL_HIDDEN_DIM", p["hidden_dim"])
    _update_config("SSL_PROJ_DIM", p["proj_dim"])
    _update_config("SSL_N_ENCODER_LAYERS", p["n_encoder_layers"])
    _update_config("SSL_DROPOUT", p["dropout"])
    _update_config("SSL_CORRUPTION_RATE", p["corruption_rate"])
    _update_config("SSL_TEMPERATURE", p["temperature"])
    _update_config("SSL_PRETRAIN_LR", p["pretrain_lr"])
    _update_config("SSL_PRETRAIN_EPOCHS", p["pretrain_epochs"])
    _update_config("SSL_FINETUNE_LR", p["finetune_lr"])
    _update_config("SSL_FINETUNE_EPOCHS", p["finetune_epochs"])
    _update_config("SSL_BATCH_SIZE", p["batch_size"])
    _update_config("SSL_WEIGHT_DECAY", p["weight_decay"])
    _update_config("SSL_FREEZE_ENCODER", p["freeze_encoder"])
    print(f"   SSL_HIDDEN_DIM       = {p['hidden_dim']}")
    print(f"   SSL_PROJ_DIM         = {p['proj_dim']}")
    print(f"   SSL_N_ENCODER_LAYERS = {p['n_encoder_layers']}")
    print(f"   SSL_DROPOUT          = {p['dropout']:.6f}")
    print(f"   SSL_CORRUPTION_RATE  = {p['corruption_rate']:.6f}")
    print(f"   SSL_TEMPERATURE      = {p['temperature']:.6f}")
    print(f"   SSL_PRETRAIN_LR      = {p['pretrain_lr']:.6f}")
    print(f"   SSL_PRETRAIN_EPOCHS  = {p['pretrain_epochs']}")
    print(f"   SSL_FINETUNE_LR      = {p['finetune_lr']:.6f}")
    print(f"   SSL_FINETUNE_EPOCHS  = {p['finetune_epochs']}")
    print(f"   SSL_BATCH_SIZE       = {p['batch_size']}")
    print(f"   SSL_WEIGHT_DECAY     = {p['weight_decay']:.6f}")
    print(f"   SSL_FREEZE_ENCODER   = {p['freeze_encoder']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
