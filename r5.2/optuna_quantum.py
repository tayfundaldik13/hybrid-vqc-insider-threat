"""
Optuna Hyperparameter Tuning - Hybrid Quantum VQC 
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
from src import quantum_config as qconfig
from src.data_process import prepare_cert_embeddings, create_tuning_eval_split
from src.models.quantum_vqc import HybridQuantumModel

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS           = [42]
TOTAL_POOL_SIZE = 990
TUNING_SIZE     = 90
N_CV_FOLDS      = 3
N_TRIALS        = 30


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def make_objective(pools):
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
        for seed in SEEDS:
            tune_emb, tune_lab = pools[seed]
            skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
            for train_idx, val_idx in skf.split(tune_emb, tune_lab):
                set_all_seeds(seed)

                X_train = torch.tensor(tune_emb[train_idx], dtype=torch.float32)
                X_val   = torch.tensor(tune_emb[val_idx],   dtype=torch.float32)
                y_train = torch.tensor(tune_lab[train_idx], dtype=torch.float32)
                y_val   = torch.tensor(tune_lab[val_idx],   dtype=torch.float32)

                eff_bs = max(2, min(batch_size, len(X_train) - 1))
                dl = DataLoader(TensorDataset(X_train, y_train),
                                batch_size=eff_bs, shuffle=True, drop_last=True)

                model = HybridQuantumModel(
                    input_dim=qconfig.INPUT_DIM,
                    n_qubits=n_qubits, n_layers=n_layers,
                    nn_hidden=nn_hidden, nn_depth=nn_depth,
                    pre_hidden=pre_hidden,
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                loss_fn   = nn.BCEWithLogitsLoss()

                model.train()
                for _ in range(epochs):
                    for xb, yb in dl:
                        logits = model(xb)
                        loss   = loss_fn(logits, yb.unsqueeze(1))
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

                model.eval()
                with torch.no_grad():
                    probs   = torch.sigmoid(model(X_val)).cpu().numpy().flatten()
                    preds   = (probs > 0.5).astype(float)
                    targets = y_val.cpu().numpy().flatten()

                all_f1s.append(f1_score(targets, preds, zero_division=0))

        mean_f1 = float(np.mean(all_f1s))

        dummy = HybridQuantumModel(
            input_dim=qconfig.INPUT_DIM,
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
    print("Optuna - Hybrid Quantum VQC")
    print(f"Tuning pool: {TUNING_SIZE} samples | {N_CV_FOLDS}-fold CV | {N_TRIALS} trials")
    print(f"Qubit Range: [2, 4, 5, 6, 7, 8, 9]")
    print("=" * 70)

    print("\nPreparing tuning pool...")
    pools = get_tuning_pools()

    study = optuna.create_study(
        direction="maximize",
        study_name="quantum_r52_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(pools), n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    p    = best.params
    print(f"\nBest F1: {best.value:.4f} (+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"Trainable params: {best.user_attrs.get('n_params', '?')}")
    print(f"N_QUBITS   = {p['n_qubits']}")
    print(f"N_LAYERS   = {p['n_layers']}")
    print(f"NN_HIDDEN  = {p['nn_hidden']}")
    print(f"NN_DEPTH   = {p['nn_depth']}")
    print(f"PRE_HIDDEN = {p['pre_hidden']}")

    print(f"\nTop 5 Trials:")
    trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        print(f"  #{i+1} F1={t.value:.4f} | "
              f"q={t.params.get('n_qubits')} L={t.params.get('n_layers')} "
              f"lr={t.params.get('learning_rate', 0):.5f} "
              f"bs={t.params.get('batch_size')} ep={t.params.get('epochs')} "
              f"nn_h={t.params.get('nn_hidden')} nn_d={t.params.get('nn_depth')} "
              f"pre_h={t.params.get('pre_hidden')} "
              f"params={t.user_attrs.get('n_params', '?')}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
