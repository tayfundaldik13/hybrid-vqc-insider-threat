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

HOLDOUT_TEST = os.environ.get("HOLDOUT_TEST", "0") == "1"
if HOLDOUT_TEST:
    os.environ.setdefault("TUNING_THREAT_RATIO", "0.2")

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


_HOLDOUT = {}


def get_tuning_pool():
    X, y = load_behavioral_features()
    if HOLDOUT_TEST:
        from sklearn.model_selection import StratifiedGroupKFold
        from src import data_process as _dp
        users = _dp._users_cache
        tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                      .split(np.zeros((len(y), 1)), y, users))
        _HOLDOUT["X_test"], _HOLDOUT["y_test"] = X[te], y[te]
        print(f"   HOLDOUT split by user: train side {len(tr)} rows ({int(y[tr].sum())} threat), "
              f"test side {len(te)} rows ({int(y[te].sum())} threat, rate {y[te].mean()*100:.3f}%), "
              f"shared users: {len(set(users[tr]) & set(users[te]))}")
        _dp._GROUPS_BY_LEN[len(tr)] = users[tr]
        X, y = X[tr], y[tr]
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
    print("OPTUNA XGBoost TUNING - r4.2 (50-500 pipeline)")
    print(f"   Tuning pool: {TUNING_N} samples ({float(os.environ.get('TUNING_THREAT_RATIO', '0.2'))*100:.0f}% threat)")
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
        study_name="xgboost_r4.2_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    p = study.best_params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={study.best_value:.4f} (+/-{study.best_trial.user_attrs.get('f1_std', 0):.4f})")
    print(f"\n{'HOLDOUT mode: config.py NOT updated' if HOLDOUT_TEST else 'Updating ' + CONFIG_PATH}")
    print(f"{'-'*60}")
    if not HOLDOUT_TEST:
        _update_config("XGB_N_ESTIMATORS", p["n_estimators"])
    if not HOLDOUT_TEST:
        _update_config("XGB_MAX_DEPTH", p["max_depth"])
    if not HOLDOUT_TEST:
        _update_config("XGB_LEARNING_RATE", p["learning_rate"])
    if not HOLDOUT_TEST:
        _update_config("XGB_SUBSAMPLE", p["subsample"])
    if not HOLDOUT_TEST:
        _update_config("XGB_COLSAMPLE", p["colsample_bytree"])
    if not HOLDOUT_TEST:
        _update_config("XGB_MIN_CHILD_WEIGHT", p["min_child_weight"])
    if not HOLDOUT_TEST:
        _update_config("XGB_SCALE_POS_WEIGHT", p["scale_pos_weight"])
    print(f"   XGB_N_ESTIMATORS     = {p['n_estimators']}")
    print(f"   XGB_MAX_DEPTH        = {p['max_depth']}")
    print(f"   XGB_LEARNING_RATE    = {p['learning_rate']:.6f}")
    print(f"   XGB_SUBSAMPLE        = {p['subsample']:.6f}")
    print(f"   XGB_COLSAMPLE        = {p['colsample_bytree']:.6f}")
    print(f"   XGB_MIN_CHILD_WEIGHT = {p['min_child_weight']}")
    print(f"   XGB_SCALE_POS_WEIGHT = {p['scale_pos_weight']:.4f}")
    print("=" * 60)

    if HOLDOUT_TEST:
        from sklearn.metrics import (recall_score, precision_score, roc_auc_score,
                                     matthews_corrcoef, average_precision_score, confusion_matrix)
        Xt, yt = _HOLDOUT["X_test"], _HOLDOUT["y_test"]
        print(f"\nHOLDOUT TEST - trained on the {len(y_tune)}-sample pool "
              f"({int(y_tune.sum())} threat), tested on {len(yt)} rows ({int(yt.sum())} threat)")
        rows = []
        for sd in range(SEED, SEED + 5):
            m = XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                              learning_rate=p["learning_rate"], subsample=p["subsample"],
                              colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
                              scale_pos_weight=p["scale_pos_weight"], eval_metric="logloss",
                              random_state=sd, verbosity=0)
            m.fit(X_tune, y_tune)
            pr = m.predict_proba(Xt)[:, 1]; pd_ = (pr >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, pd_, labels=[0, 1]).ravel()
            rows.append([recall_score(yt, pd_, zero_division=0), precision_score(yt, pd_, zero_division=0),
                         2*tp/max(2*tp+fp+fn, 1), roc_auc_score(yt, pr), average_precision_score(yt, pr),
                         matthews_corrcoef(yt, pd_), tp, fp])
        r = np.array(rows)
        for name, i in [("Recall", 0), ("Precision", 1), ("F1", 2), ("AUC-ROC", 3), ("AUPRC", 4), ("MCC", 5)]:
            print(f"   {name:<10}: {r[:, i].mean():.4f} (+/- {r[:, i].std():.4f})")
        print(f"   TP={r[:, 6].mean():.1f}  FP={r[:, 7].mean():.1f}  (mean over 5 model seeds; test threat rate {yt.mean()*100:.3f}%)")


if __name__ == "__main__":
    main()
