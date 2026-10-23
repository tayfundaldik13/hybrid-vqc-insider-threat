import os
import sys
import random
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src import config
from src.data_process import load_behavioral_features, create_tuning_eval_split
from src.data_process import GroupAwareStratifiedKFold as StratifiedKFold, use_full_pool, multi_pool_folds
use_full_pool()

SEED       = 42
TUNING_SEEDS = int(os.environ.get('TUNING_SEEDS', '3'))
POOLS = []
TUNING_N   = 100
N_CV_FOLDS = 3
N_TRIALS   = 30
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "config.py")


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_tuning_pool():
    X, y = load_behavioral_features()
    tuning, _ = create_tuning_eval_split(
        X, y, tuning_size=TUNING_N, seed=SEED, threat_ratio=0.4
    )
    X_tune, y_tune = tuning[0], tuning[1]
    POOLS[:] = [(X_tune, y_tune)] + [
        create_tuning_eval_split(X, y, tuning_size=TUNING_N, seed=SEED + _i, threat_ratio=0.4)[0][:2]
        for _i in range(1, TUNING_SEEDS)]
    print(f"   Tuning pools ({len(POOLS)}): threats per pool = {[int(p[1].sum()) for p in POOLS]}, size = {[len(p[1]) for p in POOLS]}")
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})")
    return X_tune, y_tune


def _update_config(key, value, quote_str=False):
    with open(CONFIG_PATH) as f:
        lines = f.readlines()
    updated = []
    found = False
    for line in lines:
        if re.match(rf"^\s*{re.escape(key)}\s*=" , line):
            updated.append(f"{key} = {value!r}\n" if quote_str else f"{key} = {value}\n")
            found = True
        else:
            updated.append(line)
    if not found:
        updated.append(f"{key} = {value!r}\n" if quote_str else f"{key} = {value}\n")
    with open(CONFIG_PATH, "w") as f:
        f.writelines(updated)


def main():
    print("=" * 60)
    print("OPTUNA TUNING - r5.2 (50-500 pipeline)")
    print(f"   Tuning pool: {TUNING_N} samples (20% threat / 80% normal)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Objective: macro-F1")
    print(f"   Config target: {CONFIG_PATH}")
    print("=" * 60)

    set_all_seeds(SEED)
    X_tune, y_tune = get_tuning_pool()

    def objective(trial):
        n_estimators      = trial.suggest_int("n_estimators", 10, 300)
        max_depth         = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 30)
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        class_weight      = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])

        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s = []
        for X_tune, y_tune, tr_idx, vl_idx in multi_pool_folds(POOLS, N_CV_FOLDS, SEED, StratifiedKFold):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X_tune[tr_idx], y_tune[tr_idx])
            preds = model.predict(X_tune[vl_idx])
            f1s.append(f1_score(y_tune[vl_idx], preds, average="macro", zero_division=0))

        trial.set_user_attr("f1_std", float(np.std(f1s)))

        return float(np.mean(f1s))

    study = optuna.create_study(
        direction="maximize",
        study_name="rf_r5.2_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    p = study.best_params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={study.best_value:.4f} (+/-{study.best_trial.user_attrs.get('f1_std', 0):.4f})")
    print(f"\nUpdating {CONFIG_PATH}")
    print(f"{'-'*60}")
    _update_config("RF_N_ESTIMATORS", p["n_estimators"])
    _update_config("RF_MAX_DEPTH", p["max_depth"])
    _update_config("RF_MIN_SAMPLES_SPLIT", p["min_samples_split"])
    _update_config("RF_MIN_SAMPLES_LEAF", p["min_samples_leaf"])
    _update_config("RF_MAX_FEATURES", p["max_features"], quote_str=True)
    _update_config("RF_CLASS_WEIGHT", p["class_weight"], quote_str=True)
    print(f"   RF_N_ESTIMATORS      = {p['n_estimators']}")
    print(f"   RF_MAX_DEPTH         = {p['max_depth']}")
    print(f"   RF_MIN_SAMPLES_SPLIT = {p['min_samples_split']}")
    print(f"   RF_MIN_SAMPLES_LEAF  = {p['min_samples_leaf']}")
    print(f"   RF_MAX_FEATURES      = {p['max_features']!r}")
    print(f"   RF_CLASS_WEIGHT      = {p['class_weight']!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
