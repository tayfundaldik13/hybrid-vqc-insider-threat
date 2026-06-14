"""
Optuna Hyperparameter Tuning — BiLSTM (Singh) 
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import spedia_config as cfg
from data_process_spedia import prepare_spedia_embeddings, create_tuning_eval_split
from src.models.bilstm_classifier import BiLSTMClassifier

try:
    import optuna
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_SEED = 42
TUNING_SIZE = 90
N_CV_FOLDS  = 3
N_TRIALS    = 30


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_eval(lr, batch_size, epochs, X_train, y_train, X_val, device):
    model = BiLSTMClassifier(hidden_size=cfg.BILSTM_HIDDEN,
                              num_layers=cfg.BILSTM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    dl  = DataLoader(TensorDataset(X_train, y_t),
                     batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            preds = model(xb); loss = loss_fn(preds, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = model(X_val).cpu().numpy()
    return (probs >= 0.5).astype(float), probs


def main():
    device = cfg.DEVICE
    full_emb, full_lab, full_txt = prepare_spedia_embeddings(cfg.SPEDIA_FEATURES_PATH)
    tuning, _ = create_tuning_eval_split(
        full_emb, full_lab, tuning_size=TUNING_SIZE,
        seed=TUNING_SEED, texts=full_txt, threat_ratio=cfg.THREAT_RATIO
    )
    tune_emb, tune_lab = tuning[0], tuning[1]
    print(f"Tuning pool: {len(tune_emb)} samples "
          f"(threat={tune_lab.sum()}, normal={(tune_lab==0).sum()})")

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        epochs     = trial.suggest_int("epochs", 20, 150, step=10)

        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=TUNING_SEED)
        scores = []
        for train_idx, val_idx in cv.split(tune_emb, tune_lab):
            set_all_seeds(TUNING_SEED)
            X_tr = torch.tensor(tune_emb[train_idx], dtype=torch.float32).to(device)
            X_vl = torch.tensor(tune_emb[val_idx],   dtype=torch.float32).to(device)
            preds, _ = train_eval(lr, batch_size, epochs,
                                  X_tr, tune_lab[train_idx], X_vl, device)
            scores.append(f1_score(tune_lab[val_idx], preds, zero_division=0))
        return float(np.mean(scores))

    set_all_seeds(TUNING_SEED)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=TUNING_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest: lr={best['lr']:.6f}  batch={best['batch_size']}  "
          f"epochs={best['epochs']}  (F1={study.best_value:.4f})")
    print(f"  BILSTM_LR    = {best['lr']:.6f}")
    print(f"  BILSTM_BATCH = {best['batch_size']}")
    print(f"  BILSTM_EPOCHS = {best['epochs']}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
