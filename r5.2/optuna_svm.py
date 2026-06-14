"""
Optuna Hyperparameter Tuning - SVM 
"""
import os, sys, random
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_process import prepare_cert_embeddings, create_tuning_eval_split

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


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


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
        C      = trial.suggest_float("C", 1e-3, 100.0, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        gamma  = trial.suggest_categorical("gamma", ["scale", "auto"])

        all_f1s = []
        for seed in SEEDS:
            tune_emb, tune_lab = pools[seed]
            skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
            for train_idx, val_idx in skf.split(tune_emb, tune_lab):
                set_all_seeds(seed)
                model = SVC(C=C, kernel=kernel, gamma=gamma,
                            probability=True, random_state=seed)
                model.fit(tune_emb[train_idx], tune_lab[train_idx])
                probs = model.predict_proba(tune_emb[val_idx])[:, 1]
                preds = (probs >= 0.5).astype(float)
                all_f1s.append(f1_score(tune_lab[val_idx], preds, zero_division=0))
        return float(np.mean(all_f1s))
    return objective


def main():
    print("=" * 60)
    print("Optuna — SVM")
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
    print(f"\nBest: C={best['C']:.6f}  kernel={best['kernel']}  gamma={best['gamma']}  "
          f"(F1={study.best_value:.4f})")
    print(f"SVM_C      = {best['C']:.6f}")
    print(f"SVM_KERNEL = \"{best['kernel']}\"")
    print(f"SVM_GAMMA  = \"{best['gamma']}\"")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
