"""
Optuna Hyperparameter Tuning — SVM
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import spedia_config as cfg
from data_process_spedia import prepare_spedia_embeddings, create_tuning_eval_split

try:
    import optuna
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_SEED = 42
TUNING_SIZE = 90
N_CV_FOLDS  = 3
N_TRIALS    = 50


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    full_emb, full_lab, full_txt = prepare_spedia_embeddings(cfg.SPEDIA_FEATURES_PATH)
    tuning, _ = create_tuning_eval_split(
        full_emb, full_lab, tuning_size=TUNING_SIZE,
        seed=TUNING_SEED, texts=full_txt, threat_ratio=cfg.THREAT_RATIO
    )
    tune_emb, tune_lab = tuning[0], tuning[1]
    print(f"Tuning pool: {len(tune_emb)} samples "
          f"(threat={tune_lab.sum()}, normal={(tune_lab==0).sum()})")

    def objective(trial):
        C      = trial.suggest_float("C", 1e-3, 100.0, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        gamma  = trial.suggest_categorical("gamma", ["scale", "auto"])

        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=TUNING_SEED)
        scores = []
        for train_idx, val_idx in cv.split(tune_emb, tune_lab):
            model = SVC(C=C, kernel=kernel, gamma=gamma,
                        probability=True, random_state=TUNING_SEED)
            model.fit(tune_emb[train_idx], tune_lab[train_idx])
            probs = model.predict_proba(tune_emb[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(float)
            scores.append(f1_score(tune_lab[val_idx], preds, zero_division=0))
        return float(np.mean(scores))

    set_all_seeds(TUNING_SEED)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=TUNING_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest: C={best['C']:.6f}  kernel={best['kernel']}  gamma={best['gamma']}  "
          f"(F1={study.best_value:.4f})")
    print(f"  SVM_C      = {best['C']:.6f}")
    print(f"  SVM_KERNEL = \"{best['kernel']}\"")
    print(f"  SVM_GAMMA  = \"{best['gamma']}\"")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
