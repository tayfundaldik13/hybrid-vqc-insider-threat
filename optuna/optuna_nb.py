"""
Optuna Hyperparameter Tuning for Naive Bayes
"""
import os
import sys
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_process import prepare_cert_embeddings, create_tuning_eval_split

try:
    import optuna
except ImportError:
    print("Installing optuna...")
    os.system(f"{sys.executable} -m pip install optuna")
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
        var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True)

        all_f1s = []
        for seed in SEEDS:
            tune_emb, tune_lab = tuning_pools[seed]
            skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)

            for train_idx, val_idx in skf.split(tune_emb, tune_lab):
                set_all_seeds(seed)

                X_train = tune_emb[train_idx]
                X_val   = tune_emb[val_idx]
                y_train = tune_lab[train_idx]
                y_val   = tune_lab[val_idx]

                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                f1 = f1_score(y_val, preds, zero_division=0)
                all_f1s.append(f1)

        mean_f1 = np.mean(all_f1s)
        trial.set_user_attr("f1_std", float(np.std(all_f1s)))
        return mean_f1

    return objective


def main():
    print("=" * 70)
    print("Optuna - Naive Bayes")
    print(f"Tuning: var_smoothing (log scale 1e-12 to 1e-1)")
    print(f"Tuning pool: {TUNING_SIZE} samples (isolated)")
    print(f"Seeds: {SEEDS} x {N_CV_FOLDS}-fold CV = {len(SEEDS) * N_CV_FOLDS} evals/trial")
    print(f"Trials: {N_TRIALS}")
    print("=" * 70)

    print(f"\nPreparing tuning pools ({TUNING_SIZE} samples x {len(SEEDS)} seeds)...")
    tuning_pools = get_tuning_pools()

    print(f"\n{'=' * 70}")
    print(f"Running Optuna ({N_TRIALS} trials)...")
    print(f"{'=' * 70}")

    study = optuna.create_study(
        direction="maximize",
        study_name="nb_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(make_objective(tuning_pools), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'=' * 70}")
    print("Best Trial")
    print(f"{'=' * 70}")
    best = study.best_trial
    print(f"F1 (mean across {len(SEEDS)}seeds x {N_CV_FOLDS}folds): {best.value:.4f} "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")

    print(f"{'─' * 70}")
    p = best.params
    print(f"VAR_SMOOTHING = {p['var_smoothing']:.2e}")
    print(f"\n{'─' * 70}")
    print("Top 5 Trials:")
    print(f"{'─' * 70}")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        std = t.user_attrs.get("f1_std", 0)
        print(f"  #{i+1} F1={t.value:.4f}+/-{std:.4f} | "
              f"var_smoothing={t.params.get('var_smoothing', 0):.2e}")


if __name__ == "__main__":
    main()
