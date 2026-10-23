import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

_ALL_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR     = os.path.dirname(_ALL_DATA_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _ALL_DATA_DIR)

from src.data_process import load_behavioral_features
from src.data_process import use_tuning_pool
use_tuning_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'config.py'))

try:
    import optuna
except ImportError:
    os.system(f"{sys.executable} -m pip install optuna")
    import optuna

try:
    from xgboost import XGBClassifier
except ImportError:
    os.system(f"{sys.executable} -m pip install xgboost")
    from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED       = 42
TUNING_N   = 3000
N_CV_FOLDS = 3
N_TRIALS   = 20


def get_tuning_pool():
    X, y = load_behavioral_features()
    np.random.seed(SEED)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    real_ratio = y.mean()
    n_pos = max(1, int(TUNING_N * real_ratio))
    n_neg = TUNING_N - n_pos

    sel = np.concatenate([
        np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
    ])
    np.random.shuffle(sel)

    X_tune, y_tune = X[sel], y[sel]
    spw = float((y_tune == 0).sum() / max((y_tune == 1).sum(), 1))
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})")
    print(f"   scale_pos_weight baseline: {spw:.2f}")
    return X_tune, y_tune, spw


def main():
    print("=" * 60)
    print("OPTUNA XGBoost TUNING - REAL CLASS DISTRIBUTION")
    print(f"   Tuning pool: {TUNING_N} samples at real ratio (~{int(TUNING_N*0.0468)} threat)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Objective: macro-F1  |  scale_pos_weight in search space")
    print("=" * 60)

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
        for tr_idx, vl_idx in skf.split(X_tune, y_tune):
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
        study_name="xgboost_all_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    p    = best.params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"\nUpdate in all_data/src/config.py:")
    print(f"{'-'*60}")
    print(f"XGB_N_ESTIMATORS     = {p['n_estimators']}")
    print(f"XGB_MAX_DEPTH        = {p['max_depth']}")
    print(f"XGB_LEARNING_RATE    = {p['learning_rate']:.6f}")
    print(f"XGB_SUBSAMPLE        = {p['subsample']:.6f}")
    print(f"XGB_COLSAMPLE        = {p['colsample_bytree']:.6f}")
    print(f"XGB_MIN_CHILD_WEIGHT = {p['min_child_weight']}")
    print(f"XGB_SCALE_POS_WEIGHT = {p['scale_pos_weight']:.4f}")

    print(f"\nTOP 5 TRIALS:")
    trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        print(f"  #{i+1} MacroF1={t.value:.4f}+/-{t.user_attrs.get('f1_std',0):.4f} | "
              f"n_est={t.params['n_estimators']} depth={t.params['max_depth']} "
              f"lr={t.params['learning_rate']:.4f} spw={t.params['scale_pos_weight']:.2f}")


if __name__ == "__main__":
    main()
