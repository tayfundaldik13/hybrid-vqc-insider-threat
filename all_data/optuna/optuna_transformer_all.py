import os
import sys
import random
import math
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
    os.system(f"{sys.executable} -m pip install optuna"); import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED       = 42
TUNING_N   = 3000
N_CV_FOLDS = 3
N_TRIALS   = 20
INPUT_DIM  = 16


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def get_tuning_pool():
    X, y = load_behavioral_features()
    np.random.seed(SEED)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    n_pos = max(1, int(TUNING_N * y.mean())); n_neg = TUNING_N - n_pos
    sel = np.concatenate([
        np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)])
    np.random.shuffle(sel)
    X_tune, y_tune = X[sel], y[sel]
    pos_w = float((y_tune == 0).sum() / max((y_tune == 1).sum(), 1))
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})  pos_weight={pos_w:.2f}")
    return X_tune, y_tune, pos_w


class BehaviorTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout, dim_feedforward):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)

        self.pos_emb = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout  = nn.Dropout(dropout)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x) + self.pos_emb
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)


def main():
    print("=" * 60)
    print("OPTUNA Transformer TUNING - REAL CLASS DISTRIBUTION")
    print(f"   Tuning pool: {TUNING_N} samples (~{int(TUNING_N*0.0468)} threat)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Architecture: 16 features as tokens -> TransformerEncoder")
    print("=" * 60)

    X_tune, y_tune, pos_weight_val = get_tuning_pool()
    device = config.DEVICE

    def objective(trial):
        d_model         = trial.suggest_categorical("d_model", [16, 32, 64])
        n_heads         = trial.suggest_categorical("n_heads", [1, 2, 4])
        n_layers        = trial.suggest_int("n_layers", 1, 4)
        dropout         = trial.suggest_float("dropout", 0.1, 0.4)
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [32, 64, 128])
        lr              = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size      = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs          = trial.suggest_categorical("epochs", [50, 100, 150])
        weight_decay    = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        if d_model % n_heads != 0:
            return 0.0

        pos_w   = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        skf     = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s     = []

        for tr_idx, vl_idx in skf.split(X_tune, y_tune):
            set_all_seeds(SEED)
            X_tr = torch.tensor(X_tune[tr_idx], dtype=torch.float32).to(device)
            X_vl = torch.tensor(X_tune[vl_idx], dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_tune[tr_idx], dtype=torch.float32).to(device)
            y_vl = y_tune[vl_idx]

            model = BehaviorTransformer(INPUT_DIM, d_model, n_heads, n_layers,
                                         dropout, dim_feedforward).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                           weight_decay=weight_decay)

            eff_bs = max(2, min(batch_size, len(X_tr) - 1))
            dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=eff_bs,
                             shuffle=True, drop_last=True)
            model.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    optimizer.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(X_vl)).cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            f1s.append(f1_score(y_vl, preds, average="macro", zero_division=0))

        dummy = BehaviorTransformer(INPUT_DIM, d_model, n_heads, n_layers,
                                     dropout, dim_feedforward)
        trial.set_user_attr("n_params", sum(p.numel() for p in dummy.parameters()))
        trial.set_user_attr("f1_std", float(np.std(f1s)))
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize",
                                 study_name="transformer_all_tuning",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial; p = best.params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"   n_params={best.user_attrs.get('n_params','?')}")
    print(f"\nUpdate in all_data/src/config.py:")
    print(f"{'-'*60}")
    print(f"TF_D_MODEL         = {p['d_model']}")
    print(f"TF_N_HEADS         = {p['n_heads']}")
    print(f"TF_N_LAYERS        = {p['n_layers']}")
    print(f"TF_DROPOUT         = {p['dropout']:.6f}")
    print(f"TF_DIM_FEEDFORWARD = {p['dim_feedforward']}")
    print(f"TF_LR              = {p['lr']:.6f}")
    print(f"TF_BATCH_SIZE      = {p['batch_size']}")
    print(f"TF_EPOCHS          = {p['epochs']}")
    print(f"TF_WEIGHT_DECAY    = {p['weight_decay']:.6f}")


if __name__ == "__main__":
    main()
