"""
Optuna Hyperparameter Tuning — Hybrid Quantum VQC
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
from src.models.quantum_vqc import HybridQuantumModel

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_SEED    = 42
TUNING_SIZE    = 90
N_CV_FOLDS     = 3
N_TRIALS       = 30


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tuning_pool():
    full_emb, full_lab, full_txt = prepare_spedia_embeddings(cfg.SPEDIA_FEATURES_PATH)
    tuning, _ = create_tuning_eval_split(
        full_emb, full_lab, tuning_size=TUNING_SIZE,
        seed=TUNING_SEED, texts=full_txt, threat_ratio=cfg.THREAT_RATIO
    )
    tune_emb, tune_lab = tuning[0], tuning[1]
    print(f"Tuning pool: {len(tune_emb)} samples "
          f"(threat={int(tune_lab.sum())}, normal={int((tune_lab==0).sum())})")
    return tune_emb, tune_lab


def make_objective(tune_emb, tune_lab):
    def objective(trial):
        n_qubits   = trial.suggest_categorical("n_qubits",   [2, 4, 5, 6, 7, 8, 9])
        n_layers   = trial.suggest_int("n_layers", 1, 5)
        lr         = trial.suggest_float("learning_rate", 1e-4, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [5, 8, 10, 16])
        epochs     = trial.suggest_categorical("epochs", [20, 40, 60])
        nn_hidden  = trial.suggest_categorical("nn_hidden",  [8, 16, 32])
        nn_depth   = trial.suggest_categorical("nn_depth",   [1, 2])
        pre_hidden = trial.suggest_categorical("pre_hidden", [0, 32, 64, 128])

        all_f1s = []

        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=TUNING_SEED)
        for train_idx, val_idx in skf.split(tune_emb, tune_lab):
            set_all_seeds(TUNING_SEED)

            X_train = torch.tensor(tune_emb[train_idx], dtype=torch.float32)
            X_val   = torch.tensor(tune_emb[val_idx],   dtype=torch.float32)
            y_train = torch.tensor(tune_lab[train_idx], dtype=torch.float32)
            y_val   = torch.tensor(tune_lab[val_idx],   dtype=torch.float32)

            effective_bs = max(2, min(batch_size, len(X_train) - 1))
            train_dl = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=effective_bs, shuffle=True, drop_last=True
            )

            model = HybridQuantumModel(
                input_dim=cfg.INPUT_DIM,
                n_qubits=n_qubits, n_layers=n_layers,
                nn_hidden=nn_hidden, nn_depth=nn_depth,
                pre_hidden=pre_hidden,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            loss_fn   = nn.BCEWithLogitsLoss()

            for _ in range(epochs):
                model.train()
                for xb, yb in train_dl:
                    logits = model(xb)
                    loss   = loss_fn(logits, yb.unsqueeze(1))
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                logits  = model(X_val)
                probs   = torch.sigmoid(logits).cpu().numpy().flatten()
                preds   = (probs > 0.5).astype(float)
                targets = y_val.cpu().numpy().flatten()

            all_f1s.append(f1_score(targets, preds, zero_division=0))

        mean_f1 = np.mean(all_f1s)

        set_all_seeds(TUNING_SEED)
        dummy = HybridQuantumModel(
            input_dim=cfg.INPUT_DIM,
            n_qubits=n_qubits, n_layers=n_layers,
            nn_hidden=nn_hidden, nn_depth=nn_depth,
            pre_hidden=pre_hidden,
        )
        trial.set_user_attr("n_params", sum(p.numel() for p in dummy.parameters()))
        trial.set_user_attr("amp_dim",  2 ** n_qubits)
        trial.set_user_attr("f1_std",   float(np.std(all_f1s)))

        return mean_f1

    return objective


def main():
    print("=" * 70)
    print("Hybrid Quantum VQC - Optuna Tuning (SPEDIA)")
    print(f"Tuning pool: {TUNING_SIZE} samples")
    print(f"{N_CV_FOLDS}-fold CV = {N_CV_FOLDS} evals/trial")
    print(f"Trials: {N_TRIALS}")
    print(f"Qubit Range: [2, 4, 5, 6, 7, 8, 9]")
    print(f"Pre-hidden Layer Neuron Range: [0, 32, 64, 128]")
    print("=" * 70)

    print(f"\nPreparing tuning pool ({TUNING_SIZE} samples)...")
    tune_emb, tune_lab = get_tuning_pool()

    print(f"\n{'=' * 70}")
    print(f"Running Optuna...")
    print(f"{'=' * 70}")

    study = optuna.create_study(
        direction="maximize",
        study_name="quantum_spedia_tuning",
        sampler=optuna.samplers.TPESampler(seed=TUNING_SEED),
    )
    study.optimize(make_objective(tune_emb, tune_lab),
                   n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'=' * 70}")
    print("Best Trial")
    print(f"{'=' * 70}")
    best = study.best_trial
    print(f"F1 (mean {N_CV_FOLDS}-fold CV): {best.value:.4f} "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"Trainable params: {best.user_attrs.get('n_params', '?')}")
    print(f"Amplitude dim: {best.user_attrs.get('amp_dim', '?')}")

    p = best.params
    print(f"N_QUBITS   = {p['n_qubits']}")
    print(f"N_LAYERS   = {p['n_layers']}")
    print(f"NN_HIDDEN  = {p['nn_hidden']}")
    print(f"NN_DEPTH   = {p['nn_depth']}")
    print(f"PRE_HIDDEN = {p['pre_hidden']}")

    print(f"\n{'─' * 70}")
    print("Top 5 Trials:")
    print(f"{'─' * 70}")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        print(f"  #{i+1} F1={t.value:.4f}+/-{t.user_attrs.get('f1_std', 0):.4f} | "
              f"q={t.params.get('n_qubits')} "
              f"L={t.params.get('n_layers')} "
              f"lr={t.params.get('learning_rate', 0):.5f} "
              f"bs={t.params.get('batch_size')} "
              f"ep={t.params.get('epochs')} "
              f"nn_h={t.params.get('nn_hidden')} "
              f"nn_d={t.params.get('nn_depth')} "
              f"pre_h={t.params.get('pre_hidden')} "
              f"params={t.user_attrs.get('n_params', '?')}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
