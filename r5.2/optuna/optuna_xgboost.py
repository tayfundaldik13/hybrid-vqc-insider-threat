import os
import sys
import random
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src import config
from src.data_process import load_behavioral_features, create_tuning_eval_split
from src.data_process import GroupAwareStratifiedKFold as StratifiedKFold, use_full_pool, multi_pool_folds
use_full_pool()

try:
    from xgboost import XGBClassifier
except ImportError:
    os.system(f"{sys.executable} -m pip install xgboost")
    from xgboost import XGBClassifier

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
    spw = float((y_tune == 0).sum() / max((y_tune == 1).sum(), 1))
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})")
    print(f"   scale_pos_weight baseline: {spw:.2f}")
    return X_tune, y_tune, spw


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
    X_tune, y_tune, base_spw = get_tuning_pool()

    def objective(trial):
        n_estimators      = trial.suggest_int("n_estimators", 50, 500)
        max_depth         = trial.suggest_int("max_depth", 3, 10)
        learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        subsample         = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 20)
        scale_pos_weight  = trial.suggest_float("scale_pos_weight",
                                                 max(1.0, base_spw * 0.5),
                                                 base_spw * 2.0)

        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s = []
        for X_tune, y_tune, tr_idx, vl_idx in multi_pool_folds(POOLS, N_CV_FOLDS, SEED, StratifiedKFold):
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=SEED,
                verbosity=0,
            )
            model.fit(X_tune[tr_idx], y_tune[tr_idx])
            probs = model.predict_proba(X_tune[vl_idx])[:, 1]
            preds = (probs >= 0.5).astype(float)
            f1s.append(f1_score(y_tune[vl_idx], preds, average="macro", zero_division=0))

        trial.set_user_attr("f1_std", float(np.std(f1s)))

        return float(np.mean(f1s))

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_r5.2_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    p = study.best_params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={study.best_value:.4f} (+/-{study.best_trial.user_attrs.get('f1_std', 0):.4f})")
    print(f"\nUpdating {CONFIG_PATH}")
    print(f"{'-'*60}")
    _update_config("XGB_N_ESTIMATORS", p["n_estimators"])
    _update_config("XGB_MAX_DEPTH", p["max_depth"])
    _update_config("XGB_LEARNING_RATE", p["learning_rate"])
    _update_config("XGB_SUBSAMPLE", p["subsample"])
    _update_config("XGB_COLSAMPLE", p["colsample_bytree"])
    _update_config("XGB_MIN_CHILD_WEIGHT", p["min_child_weight"])
    _update_config("XGB_SCALE_POS_WEIGHT", p["scale_pos_weight"])
    print(f"   XGB_N_ESTIMATORS     = {p['n_estimators']}")
    print(f"   XGB_MAX_DEPTH        = {p['max_depth']}")
    print(f"   XGB_LEARNING_RATE    = {p['learning_rate']:.6f}")
    print(f"   XGB_SUBSAMPLE        = {p['subsample']:.6f}")
    print(f"   XGB_COLSAMPLE        = {p['colsample_bytree']:.6f}")
    print(f"   XGB_MIN_CHILD_WEIGHT = {p['min_child_weight']}")
    print(f"   XGB_SCALE_POS_WEIGHT = {p['scale_pos_weight']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
