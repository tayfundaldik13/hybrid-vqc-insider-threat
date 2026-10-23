import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config
from src.models.bilstm_classifier import BiLSTMClassifier
from src.data_process import load_behavioral_features
from src.data_process import use_tuning_pool
use_tuning_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'config.py'), rename={'LR': 'BILSTM_LR', 'HIDDEN_SIZE': 'BILSTM_HIDDEN_SIZE', 'NUM_LAYERS': 'BILSTM_NUM_LAYERS', 'BATCH_SIZE': 'BILSTM_BATCH_SIZE', 'EPOCHS': 'BILSTM_EPOCHS'})

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

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    real_ratio = y.mean()
    n_pos = max(1, int(TUNING_N * real_ratio))
    n_neg = TUNING_N - n_pos

    sel = np.concatenate([
        np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
    ])
    np.random.shuffle(sel)

    X_tune, y_tune = X[sel], y[sel]
    pos_weight_val = (y_tune == 0).sum() / max((y_tune == 1).sum(), 1)
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})")
    print(f"   pos_weight for weighted BCE: {pos_weight_val:.2f}")
    return X_tune, y_tune, float(pos_weight_val)


def make_objective(X_tune, y_tune, pos_weight_val):
    device = config.DEVICE

    def objective(trial):
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        num_layers  = trial.suggest_int("num_layers", 1, 3)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs      = trial.suggest_categorical("epochs", [50, 100, 150])

        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s = []
        pos_w = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

        for tr_idx, vl_idx in skf.split(X_tune, y_tune):
            set_all_seeds(SEED)

            X_tr = torch.tensor(X_tune[tr_idx], dtype=torch.float32).to(device)
            X_vl = torch.tensor(X_tune[vl_idx], dtype=torch.float32).to(device)
            y_tr = y_tune[tr_idx]
            y_vl = y_tune[vl_idx]

            model     = BiLSTMClassifier(hidden_size=hidden_size,
                                         num_layers=num_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn   = nn.BCELoss(weight=pos_w.expand(batch_size))

            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
            dl = DataLoader(TensorDataset(X_tr, y_tr_t),
                            batch_size=batch_size, shuffle=True, drop_last=False)

            model.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    w = pos_w.expand(len(yb))
                    _loss_fn = nn.BCELoss(weight=w)
                    preds_b  = model(xb)
                    loss     = _loss_fn(preds_b, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = model(X_vl).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(float)
            f1s.append(f1_score(y_vl, preds, average="macro", zero_division=0))

        trial.set_user_attr("f1_std", float(np.std(f1s)))
        return float(np.mean(f1s))

    return objective


def main():
    print("=" * 60)
    print("OPTUNA BiLSTM TUNING - REAL CLASS DISTRIBUTION")
    print(f"   Tuning pool: {TUNING_N} samples at real ratio (~{int(TUNING_N*0.0468)} threat)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Objective: macro-F1  |  Weighted BCE for imbalance")
    print("=" * 60)

    X_tune, y_tune, pos_weight_val = get_tuning_pool()

    study = optuna.create_study(
        direction="maximize",
        study_name="singh_all_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(make_objective(X_tune, y_tune, pos_weight_val),
                   n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    p    = best.params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std',0):.4f})")
    print(f"\nUpdate in all_data/main_singh_all.py:")
    print(f"{'-'*60}")
    print(f"LR          = {p['lr']:.6f}")
    print(f"HIDDEN_SIZE = {p['hidden_size']}")
    print(f"NUM_LAYERS  = {p['num_layers']}")
    print(f"BATCH_SIZE  = {p['batch_size']}")
    print(f"EPOCHS      = {p['epochs']}")

    print(f"\nTOP 5 TRIALS:")
    trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        print(f"  #{i+1} MacroF1={t.value:.4f}+/-{t.user_attrs.get('f1_std',0):.4f} | "
              f"lr={t.params['lr']:.5f} hidden={t.params['hidden_size']} "
              f"layers={t.params['num_layers']} bs={t.params['batch_size']} "
              f"ep={t.params['epochs']}")


if __name__ == "__main__":
    main()
