"""
Optuna Hyperparameter Tuning for Hybrid Quantum VQC
"""
import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src import quantum_config as qconfig
from src.models.quantum_vqc import HybridQuantumModel
from src.data_process import prepare_cert_embeddings, create_tuning_eval_split

try:
    import optuna
except ImportError:
    print("Installing optuna...")
    os.system(f"{sys.executable} -m pip install optuna")
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS = [42]
TOTAL_POOL_SIZE = 990      
TUNING_SIZE = 90           
N_CV_FOLDS = 3
N_TRIALS = 30


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
        full_emb, full_lab, full_txt = prepare_cert_embeddings(
            max_size=TOTAL_POOL_SIZE, seed=seed
        )
        tuning, _ = create_tuning_eval_split(
            full_emb, full_lab, tuning_size=TUNING_SIZE, seed=seed, texts=full_txt
        )
        tune_emb, tune_lab, _ = tuning
        tuning_pools[seed] = (tune_emb, tune_lab)
        print(f"Seed {seed}: tuning={len(tune_emb)} "
              f"(threat={int(tune_lab.sum())}, normal={int((tune_lab == 0).sum())})")
    return tuning_pools


def make_objective(tuning_pools):
    def objective(trial):
        n_qubits = trial.suggest_categorical("n_qubits", [2, 4, 5, 6, 7, 8, 9])
        n_layers = trial.suggest_int("n_layers", 1, 5)
        lr = trial.suggest_float("learning_rate", 1e-4, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [5, 8, 10, 16])
        epochs = trial.suggest_categorical("epochs", [20, 40, 60])
        nn_hidden = trial.suggest_categorical("nn_hidden", [8, 16, 32])
        nn_depth = trial.suggest_categorical("nn_depth", [1, 2])
        pre_hidden = trial.suggest_categorical("pre_hidden", [0, 32, 64, 128])

        all_f1s = []

        for seed in SEEDS:
            tune_emb, tune_lab = tuning_pools[seed]

            skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)

            for train_idx, val_idx in skf.split(tune_emb, tune_lab):
                set_all_seeds(seed)

                X_train = torch.tensor(tune_emb[train_idx], dtype=torch.float32)
                X_val = torch.tensor(tune_emb[val_idx], dtype=torch.float32)
                y_train = torch.tensor(tune_lab[train_idx], dtype=torch.float32)
                y_val = torch.tensor(tune_lab[val_idx], dtype=torch.float32)

                effective_bs = min(batch_size, len(X_train) - 1)
                if effective_bs < 2:
                    effective_bs = 2

                train_dl = DataLoader(
                    TensorDataset(X_train, y_train),
                    batch_size=effective_bs, shuffle=True, drop_last=True
                )

                model = HybridQuantumModel(
                    input_dim=qconfig.INPUT_DIM,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    nn_hidden=nn_hidden,
                    nn_depth=nn_depth,
                    pre_hidden=pre_hidden,
                )

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                loss_fn = nn.BCEWithLogitsLoss()

                for epoch in range(epochs):
                    model.train()
                    for xb, yb in train_dl:
                        logits = model(xb)
                        loss = loss_fn(logits, yb.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    logits = model(X_val)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(float)
                    targets = y_val.cpu().numpy().flatten()

                f1 = f1_score(targets, preds, zero_division=0)
                all_f1s.append(f1)

        mean_f1 = np.mean(all_f1s)

        set_all_seeds(42)
        dummy = HybridQuantumModel(
            input_dim=qconfig.INPUT_DIM,
            n_qubits=n_qubits, n_layers=n_layers,
            nn_hidden=nn_hidden, nn_depth=nn_depth,
            pre_hidden=pre_hidden,
        )
        n_params = sum(p.numel() for p in dummy.parameters())
        trial.set_user_attr("n_params", n_params)
        trial.set_user_attr("amp_dim", 2 ** n_qubits)
        trial.set_user_attr("f1_std", float(np.std(all_f1s)))

        return mean_f1

    return objective


def main():
    print("=" * 70)
    print("Optuna — Hybrid Quantum VQC")
    print(f"Architecture: Pre-Quantum → AmplitudeEmbedding → StronglyEntanglingLayers → Post-Quantum")
    print(f"Tuning pool: {TUNING_SIZE} samples (ISOLATED — never seen during evaluation)")
    print(f"Seeds: {SEEDS} × {N_CV_FOLDS}-fold CV = {len(SEEDS) * N_CV_FOLDS} evals/trial")
    print(f"Trials: {N_TRIALS}")
    print(f"Qubit Range: [2, 4, 5, 6, 7, 8, 9]")
    print(f"Pre-hidden Range: [0, 32, 64, 128]")
    print(f"Layer search: [1, 2, 3, 4, 5]")
    print("=" * 70)

    print(f"\nPreparing tuning pools ({TUNING_SIZE} samples × {len(SEEDS)} seeds)...")
    tuning_pools = get_tuning_pools()

    print(f"\n{'=' * 70}")
    print(f"Running Optuna ({N_TRIALS} trials)...")
    print(f"{'=' * 70}")

    study = optuna.create_study(
        direction="maximize",
        study_name="quantum_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective_fn = make_objective(tuning_pools)
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=True)

    # ── Step 3: Results ──
    print(f"\n{'=' * 70}")
    print(f"Best Trial")
    print(f"{'=' * 70}")
    best = study.best_trial
    print(f"   F1 (mean across {len(SEEDS)}seeds × {N_CV_FOLDS}folds): {best.value:.4f} "
          f"(±{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"Trainable params: {best.user_attrs.get('n_params', '?')}")
    print(f"Amplitude dim: {best.user_attrs.get('amp_dim', '?')}")

    print(f"{'─' * 70}")
    p = best.params
    print(f"N_QUBITS = {p['n_qubits']}")
    print(f"N_LAYERS = {p['n_layers']}")
    print(f"LEARNING_RATE = {p['learning_rate']:.6f}")
    print(f"BATCH_SIZE = {p['batch_size']}")
    print(f"EPOCHS = {p['epochs']}")
    print(f"NN_HIDDEN = {p['nn_hidden']}")
    print(f"NN_DEPTH = {p['nn_depth']}")
    print(f"PRE_HIDDEN = {p['pre_hidden']}")


    print(f"\n{'─' * 70}")
    print("Top 5 Trials:")
    print(f"{'─' * 70}")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        n_p = t.user_attrs.get("n_params", "?")
        std = t.user_attrs.get("f1_std", 0)
        print(f"  #{i+1} F1={t.value:.4f}±{std:.4f} | q={t.params.get('n_qubits')} "
              f"L={t.params.get('n_layers')} "
              f"lr={t.params.get('learning_rate', 0):.5f} "
              f"bs={t.params.get('batch_size')} "
              f"ep={t.params.get('epochs')} "
              f"nn_h={t.params.get('nn_hidden')} "
              f"nn_d={t.params.get('nn_depth')} "
              f"pre_h={t.params.get('pre_hidden')} "
              f"params={n_p}")


if __name__ == "__main__":
    main()
