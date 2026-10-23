import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

_ALL_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR     = os.path.dirname(_ALL_DATA_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _ALL_DATA_DIR)

from src import config
from src.data_process import load_behavioral_features
from src.data_process import use_tuning_pool
use_tuning_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'config.py'))

try:
    import optuna
except ImportError:
    os.system(f"{sys.executable} -m pip install optuna")
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED       = 42
TUNING_N   = 3000
N_CV_FOLDS = 3
N_TRIALS   = 20
INPUT_DIM  = 16


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tuning_pool():
    X, y = load_behavioral_features()
    np.random.seed(SEED)

    pos_idx    = np.where(y == 1)[0]
    neg_idx    = np.where(y == 0)[0]
    real_ratio = y.mean()
    n_pos      = max(1, int(TUNING_N * real_ratio))
    n_neg      = TUNING_N - n_pos

    sel = np.concatenate([
        np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
    ])
    np.random.shuffle(sel)

    X_tune, y_tune = X[sel], y[sel]
    pos_weight_val = float((y_tune == 0).sum() / max((y_tune == 1).sum(), 1))
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})")
    print(f"   pos_weight for BCEWithLogitsLoss: {pos_weight_val:.2f}")
    return X_tune, y_tune, pos_weight_val


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    print("=" * 60)
    print("OPTUNA MLP TUNING - REAL CLASS DISTRIBUTION")
    print(f"   Tuning pool: {TUNING_N} samples at real ratio (~{int(TUNING_N*0.0468)} threat)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Objective: macro-F1  |  pos_weight in BCEWithLogitsLoss")
    print("=" * 60)

    X_tune, y_tune, pos_weight_val = get_tuning_pool()
    device = config.DEVICE

    def objective(trial):
        hidden_size  = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        num_layers   = trial.suggest_int("num_layers", 1, 3)
        dropout      = trial.suggest_float("dropout", 0.1, 0.6)
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs       = trial.suggest_categorical("epochs", [50, 100, 150])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        pos_w = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        skf   = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s   = []

        for tr_idx, vl_idx in skf.split(X_tune, y_tune):
            set_all_seeds(SEED)

            X_tr = torch.tensor(X_tune[tr_idx], dtype=torch.float32).to(device)
            X_vl = torch.tensor(X_tune[vl_idx], dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_tune[tr_idx], dtype=torch.float32).to(device)
            y_vl = y_tune[vl_idx]

            model     = SimpleMLP(INPUT_DIM, hidden_size, num_layers, dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_w)

            eff_bs   = max(2, min(batch_size, len(X_tr) - 1))
            train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                                  batch_size=eff_bs, shuffle=True, drop_last=True)

            model.train()
            for _ in range(epochs):
                for xb, yb in train_dl:
                    logits = model(xb)
                    loss   = loss_fn(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(X_vl)).cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            f1s.append(f1_score(y_vl, preds, average="macro", zero_division=0))

        dummy = SimpleMLP(INPUT_DIM, hidden_size, num_layers, dropout)
        trial.set_user_attr("n_params", sum(p.numel() for p in dummy.parameters()))
        trial.set_user_attr("f1_std", float(np.std(f1s)))
        return float(np.mean(f1s))

    study = optuna.create_study(
        direction="maximize",
        study_name="mlp_all_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    p    = best.params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"   n_params={best.user_attrs.get('n_params','?')}")
    print(f"\nUpdate in all_data/src/config.py:")
    print(f"{'-'*60}")
    print(f"MLP_HIDDEN_SIZE  = {p['hidden_size']}")
    print(f"MLP_NUM_LAYERS   = {p['num_layers']}")
    print(f"MLP_DROPOUT      = {p['dropout']:.6f}")
    print(f"MLP_LR           = {p['lr']:.6f}")
    print(f"MLP_BATCH_SIZE   = {p['batch_size']}")
    print(f"MLP_EPOCHS       = {p['epochs']}")
    print(f"MLP_WEIGHT_DECAY = {p['weight_decay']:.6f}")

    print(f"\nTOP 5 TRIALS:")
    trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        print(f"  #{i+1} MacroF1={t.value:.4f}+/-{t.user_attrs.get('f1_std',0):.4f} | "
              f"h={t.params['hidden_size']} L={t.params['num_layers']} "
              f"lr={t.params['lr']:.5f} ep={t.params['epochs']} "
              f"params={t.user_attrs.get('n_params','?')}")


if __name__ == "__main__":
    main()
