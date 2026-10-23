import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_process import load_behavioral_features
from src.data_process import use_tuning_pool
use_tuning_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'config.py'))

try:
    import optuna
except ImportError:
    os.system(f"{sys.executable} -m pip install optuna")
    import optuna

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
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})")
    return X_tune, y_tune


def main():
    print("=" * 60)
    print("OPTUNA SVM TUNING - REAL CLASS DISTRIBUTION")
    print(f"   Tuning pool: {TUNING_N} samples at real ratio (~{int(TUNING_N*0.0468)} threat)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Objective: macro-F1  |  class_weight in search space")
    print("=" * 60)

    X_tune, y_tune = get_tuning_pool()

    def objective(trial):
        C            = trial.suggest_float("C", 1e-3, 100.0, log=True)
        kernel       = trial.suggest_categorical("kernel", ["linear", "rbf"])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

        gamma = "scale"
        if kernel == "rbf":
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s = []
        for tr_idx, vl_idx in skf.split(X_tune, y_tune):
            model = SVC(
                C=C, kernel=kernel, gamma=gamma,
                class_weight=class_weight,
                probability=True, random_state=SEED,
            )
            model.fit(X_tune[tr_idx], y_tune[tr_idx])
            probs = model.predict_proba(X_tune[vl_idx])[:, 1]
            preds = (probs >= 0.5).astype(float)
            f1s.append(f1_score(y_tune[vl_idx], preds, average="macro", zero_division=0))

        trial.set_user_attr("f1_std", float(np.std(f1s)))
        return float(np.mean(f1s))

    study = optuna.create_study(
        direction="maximize",
        study_name="svm_all_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    p    = best.params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std',0):.4f})")
    print(f"\nUpdate in all_data/main_svm_all.py:")
    print(f"{'-'*60}")
    print(f"SVM_C            = {p['C']:.6f}")
    print(f"SVM_KERNEL       = \"{p['kernel']}\"")
    print(f"SVM_GAMMA        = \"{p.get('gamma', 'scale')}\"")
    print(f"SVM_CLASS_WEIGHT = {repr(p['class_weight'])}")

    print(f"\nTOP 5 TRIALS:")
    trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        print(f"  #{i+1} MacroF1={t.value:.4f}+/-{t.user_attrs.get('f1_std',0):.4f} | "
              f"C={t.params['C']:.4f} kernel={t.params['kernel']} "
              f"class_weight={t.params['class_weight']}")


if __name__ == "__main__":
    main()
