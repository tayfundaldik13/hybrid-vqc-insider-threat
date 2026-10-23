import os, sys, random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_process import prepare_cert_embeddings, create_tuning_eval_split
from src.data_process import GroupAwareStratifiedKFold as StratifiedKFold, use_full_pool
use_full_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'src', 'config.py'))

try:
    import optuna
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS           = [42, 43, 44]
TOTAL_POOL_SIZE = 990
TUNING_SIZE     = 100
N_CV_FOLDS      = 3
N_TRIALS        = 30


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_tuning_pools():
    pools = {}
    for seed in SEEDS:
        full_emb, full_lab = prepare_cert_embeddings(
            max_size=TOTAL_POOL_SIZE, seed=seed
        )
        tuning, _ = create_tuning_eval_split(
            full_emb, full_lab, tuning_size=TUNING_SIZE, seed=seed
        )
        tune_emb, tune_lab = tuning
        pools[seed] = (tune_emb, tune_lab)
        print(f"   Seed {seed}: tuning={len(tune_emb)} "
              f"(threat={int(tune_lab.sum())}, normal={int((tune_lab == 0).sum())})")
    return pools


def make_objective(pools):
    def objective(trial):
        var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True)
        all_f1s = []
        for seed in SEEDS:
            tune_emb, tune_lab = pools[seed]
            skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
            for train_idx, val_idx in skf.split(tune_emb, tune_lab):
                set_all_seeds(seed)
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(tune_emb[train_idx], tune_lab[train_idx])
                probs = model.predict_proba(tune_emb[val_idx])[:, 1]
                preds = (probs >= 0.5).astype(float)
                all_f1s.append(f1_score(tune_lab[val_idx], preds, average="macro", zero_division=0))
        return float(np.mean(all_f1s))
    return objective


def main():
    print("=" * 60)
    print("OPTUNA - NB (CERT r5.2)")
    print(f"   Tuning pool: {TUNING_SIZE} samples (20% threat) | {N_CV_FOLDS}-fold CV | {N_TRIALS} trials")
    print("=" * 60)

    print("\nPreparing tuning pool...")
    pools = get_tuning_pools()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(pools), n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest var_smoothing: {best['var_smoothing']:.4e}  (Macro-F1={study.best_value:.4f})")
    print(f"\n-> Writing to r5.2/src/config.py:")
    print(f"VAR_SMOOTHING = {best['var_smoothing']:.6e}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
