import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.models.bilstm_classifier import BiLSTMClassifier
from src.data_process import prepare_cert_embeddings, create_tuning_eval_split
from src.data_process import GroupAwareStratifiedKFold as StratifiedKFold, use_full_pool
use_full_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'config.py'), rename={'LR': 'BILSTM_LR', 'EPOCHS': 'BILSTM_EPOCHS', 'HIDDEN_SIZE': 'BILSTM_HIDDEN', 'NUM_LAYERS': 'BILSTM_LAYERS', 'BATCH_SIZE': 'BILSTM_BATCH'})

try:
    import optuna
except ImportError:
    print("Installing optuna...")
    os.system(f"{sys.executable} -m pip install optuna")
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS = [42, 43, 44]
TOTAL_POOL_SIZE = 990
TUNING_SIZE = 100
N_CV_FOLDS = 3
N_TRIALS = 30

HIDDEN_SIZE = 256
NUM_LAYERS  = 3
BATCH_SIZE  = 32


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tuning_pools():
    tuning_pools = {}
    for seed in SEEDS:
        full_emb, full_lab = prepare_cert_embeddings(
            max_size=TOTAL_POOL_SIZE, seed=seed
        )
        tuning, _ = create_tuning_eval_split(
            full_emb, full_lab, tuning_size=TUNING_SIZE, seed=seed
        )
        tune_emb, tune_lab = tuning
        tuning_pools[seed] = (tune_emb, tune_lab)
        print(f"   Seed {seed}: tuning={len(tune_emb)} "
              f"(threat={int(tune_lab.sum())}, normal={int((tune_lab == 0).sum())})")
    return tuning_pools


def make_objective(tuning_pools):
    def objective(trial):
        lr     = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 20, 100)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        num_layers  = trial.suggest_int("num_layers", 1, 3)
        batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])

        device = config.DEVICE
        all_f1s = []

        for seed in SEEDS:
            tune_emb, tune_lab = tuning_pools[seed]
            skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)

            for train_idx, val_idx in skf.split(tune_emb, tune_lab):
                set_all_seeds(seed)

                X_train = torch.tensor(tune_emb[train_idx], dtype=torch.float32).to(device)
                X_val   = torch.tensor(tune_emb[val_idx],   dtype=torch.float32).to(device)
                y_train = torch.tensor(tune_lab[train_idx], dtype=torch.float32).to(device)
                y_val_np = tune_lab[val_idx]

                effective_bs = min(batch_size, len(X_train) - 1)
                if effective_bs < 2:
                    effective_bs = 2

                dl = DataLoader(TensorDataset(X_train, y_train),
                                batch_size=effective_bs, shuffle=True, drop_last=True)

                model = BiLSTMClassifier(
                    hidden_size=hidden_size, num_layers=num_layers
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_fn   = nn.BCELoss()

                model.train()
                for _ in range(epochs):
                    for xb, yb in dl:
                        preds = model(xb)
                        loss  = loss_fn(preds, yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    probs = model(X_val).cpu().numpy()
                preds_val = (probs >= 0.5).astype(float)
                f1 = f1_score(y_val_np, preds_val, average="macro", zero_division=0)
                all_f1s.append(f1)

        mean_f1 = np.mean(all_f1s)
        trial.set_user_attr("f1_std", float(np.std(all_f1s)))

        dummy = BiLSTMClassifier(hidden_size=hidden_size, num_layers=num_layers)
        trial.set_user_attr("n_params", sum(p.numel() for p in dummy.parameters()))

        return mean_f1

    return objective


def main():
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING - Singh et al. BiLSTM")
    print(f"   Tuning: hidden_size, num_layers, batch_size, learning_rate, epochs (paper values 256/3/32 are inside the search space)")
    print(f"   Tuning pool: {TUNING_SIZE} samples (20% threat / 80% normal)")
    print(f"   Seeds: {SEEDS} x {N_CV_FOLDS}-fold CV = {len(SEEDS) * N_CV_FOLDS} evals/trial")
    print(f"   Trials: {N_TRIALS}")
    print("=" * 70)

    print(f"\nPreparing tuning pools ({TUNING_SIZE} samples x {len(SEEDS)} seeds)...")
    tuning_pools = get_tuning_pools()

    print(f"\n{'=' * 70}")
    print(f"Running Optuna ({N_TRIALS} trials)...")
    print(f"{'=' * 70}")

    study = optuna.create_study(
        direction="maximize",
        study_name="singh_bilstm_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(make_objective(tuning_pools), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'=' * 70}")
    print("BEST TRIAL")
    print(f"{'=' * 70}")
    best = study.best_trial
    print(f"   Macro-F1 (mean across {len(SEEDS)}seeds x {N_CV_FOLDS}folds): {best.value:.4f} "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"   Trainable params: {best.user_attrs.get('n_params', '?')}")

    print(f"\nCopy-paste these into main_singh.py:")
    print(f"{'-' * 70}")
    p = best.params
    print(f"LR     = {p['learning_rate']:.6f}")
    print(f"EPOCHS = {p['epochs']}")
    print(f"HIDDEN_SIZE = {p['hidden_size']}")
    print(f"NUM_LAYERS = {p['num_layers']}")
    print(f"BATCH_SIZE = {p['batch_size']}")

    print(f"\n{'-' * 70}")
    print("TOP 5 TRIALS:")
    print(f"{'-' * 70}")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        std = t.user_attrs.get("f1_std", 0)
        n_p = t.user_attrs.get("n_params", "?")
        print(f"  #{i+1} Macro-F1={t.value:.4f}+/-{std:.4f} | "
              f"lr={t.params.get('learning_rate', 0):.6f} "
              f"ep={t.params.get('epochs')} "
              f"params={n_p}")


if __name__ == "__main__":
    main()
