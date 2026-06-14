"""
Optuna Hyperparameter Tuning — BiLSTM 
"""
import os, sys, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_process import prepare_cert_embeddings, create_tuning_eval_split
from src.models.bilstm_classifier import BiLSTMClassifier

try:
    import optuna
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS           = [42]
TOTAL_POOL_SIZE = 990
TUNING_SIZE     = 90
N_CV_FOLDS      = 3
N_TRIALS        = 30

HIDDEN_SIZE = 256
NUM_LAYERS  = 3


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tuning_pools():
    pools = {}
    for seed in SEEDS:
        full_emb, full_lab, full_txt = prepare_cert_embeddings(
            max_size=TOTAL_POOL_SIZE, seed=seed
        )
        tuning, _ = create_tuning_eval_split(
            full_emb, full_lab, tuning_size=TUNING_SIZE, seed=seed, texts=full_txt
        )
        tune_emb, tune_lab, _ = tuning
        pools[seed] = (tune_emb, tune_lab)
        print(f"Seed {seed}: tuning={len(tune_emb)} "
              f"(threat={int(tune_lab.sum())}, normal={int((tune_lab == 0).sum())})")
    return pools


def train_eval(lr, batch_size, epochs, X_train, y_train, X_val, device):
    model = BiLSTMClassifier(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    dl  = DataLoader(TensorDataset(X_train, y_t),
                     batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            preds = model(xb)
            loss  = loss_fn(preds, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = model(X_val).cpu().numpy()
    return (probs >= 0.5).astype(float), probs


def make_objective(pools):
    device = config.DEVICE

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        epochs     = trial.suggest_int("epochs", 20, 150, step=10)

        all_f1s = []
        for seed in SEEDS:
            tune_emb, tune_lab = pools[seed]
            skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
            for train_idx, val_idx in skf.split(tune_emb, tune_lab):
                set_all_seeds(seed)
                X_tr = torch.tensor(tune_emb[train_idx], dtype=torch.float32).to(device)
                X_vl = torch.tensor(tune_emb[val_idx],   dtype=torch.float32).to(device)
                preds, _ = train_eval(lr, batch_size, epochs,
                                      X_tr, tune_lab[train_idx], X_vl, device)
                all_f1s.append(f1_score(tune_lab[val_idx], preds, zero_division=0))
        return float(np.mean(all_f1s))
    return objective


def main():
    print("=" * 60)
    print("Optuna - BiLSTM")
    print(f"Tuning pool: {TUNING_SIZE} samples | {N_CV_FOLDS}-fold CV | {N_TRIALS} trials")
    print("=" * 60)

    print("\nPreparing tuning pool...")
    pools = get_tuning_pools()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(pools), n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest: lr={best['lr']:.6f}  batch={best['batch_size']}  "
          f"epochs={best['epochs']}  (F1={study.best_value:.4f})")
    print(f"LR         = {best['lr']:.6f}")
    print(f"BATCH_SIZE = {best['batch_size']}")
    print(f"EPOCHS     = {best['epochs']}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
