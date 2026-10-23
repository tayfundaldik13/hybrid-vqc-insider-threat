import os, sys, argparse, numpy as np, random, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as cfg
from src.data_process import (
    prepare_spedia_embeddings, create_nonoverlapping_stages, create_nested_folds
)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)

def save_pred(results_dir, seed, data_size, outer_idx, y_true, y_pred, test_idx):
    pred_dir = os.path.join(results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, f"seed{seed}_size{data_size}_fold{outer_idx}.npz")
    if os.path.exists(path):
        return
    np.savez_compressed(path, y_true=y_true, y_pred=y_pred, test_idx=test_idx)

def count_missing(results_dir, seeds, data_sizes, n_outer):
    pred_dir = os.path.join(results_dir, "predictions")
    missing = 0
    for seed in seeds:
        for sz in data_sizes:
            for fold in range(n_outer):
                p = os.path.join(pred_dir, f"seed{seed}_size{sz}_fold{fold}.npz")
                if not os.path.exists(p):
                    missing += 1
    return missing


def reconstruct_nb(X_all, y_all, seeds, data_sizes, n_outer):
    from sklearn.naive_bayes import GaussianNB
    total = len(seeds) * len(data_sizes) * n_outer
    done = 0
    t0 = time.time()
    print(f"\n[NB]  {count_missing(cfg.RESULTS_NB_DIR, seeds, data_sizes, n_outer)} files missing")

    for seed in seeds:
        set_seed(seed)
        stages = create_nonoverlapping_stages(
            X_all, y_all, data_sizes, seed=seed, threat_ratio=cfg.THREAT_RATIO)

        for data_size in data_sizes:
            X_s, y_s = stages[data_size]
            nested_folds = create_nested_folds(y_s, n_outer, n_inner=1, seed=seed)

            for outer_idx, fold_data in enumerate(nested_folds):
                tri = fold_data["outer_train"]
                tei = fold_data["outer_test"]
                X_tr, X_te = X_s[tri], X_s[tei]
                y_tr, y_te = y_s[tri], y_s[tei]

                model = GaussianNB(var_smoothing=cfg.VAR_SMOOTHING)
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_te)[:, 1]
                preds = (probs >= 0.5).astype(float)

                save_pred(cfg.RESULTS_NB_DIR, seed, data_size, outer_idx,
                          y_te, preds, tei)
                done += 1
                if done % 20 == 0:
                    print(f"  NB: {done}/{total} ({time.time()-t0:.1f}s)")

    n_missing = count_missing(cfg.RESULTS_NB_DIR, seeds, data_sizes, n_outer)
    print(f"  NB done. Still missing: {n_missing}")


def reconstruct_svm(X_all, y_all, seeds, data_sizes, n_outer):
    from sklearn.svm import SVC
    total = len(seeds) * len(data_sizes) * n_outer
    done = 0
    t0 = time.time()
    print(f"\n[SVM] {count_missing(cfg.RESULTS_SVM_DIR, seeds, data_sizes, n_outer)} files missing")

    for seed in seeds:
        set_seed(seed)
        stages = create_nonoverlapping_stages(
            X_all, y_all, data_sizes, seed=seed, threat_ratio=cfg.THREAT_RATIO)

        for data_size in data_sizes:
            X_s, y_s = stages[data_size]
            nested_folds = create_nested_folds(y_s, n_outer, n_inner=1, seed=seed)

            for outer_idx, fold_data in enumerate(nested_folds):
                tri = fold_data["outer_train"]
                tei = fold_data["outer_test"]
                X_tr, X_te = X_s[tri], X_s[tei]
                y_tr, y_te = y_s[tri], y_s[tei]

                model = SVC(C=cfg.SVM_C, kernel=cfg.SVM_KERNEL,
                            gamma=cfg.SVM_GAMMA, probability=True,
                            random_state=seed)
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_te)[:, 1]
                preds = (probs >= 0.5).astype(float)

                save_pred(cfg.RESULTS_SVM_DIR, seed, data_size, outer_idx,
                          y_te, preds, tei)
                done += 1
                if done % 10 == 0:
                    print(f"  SVM: {done}/{total} ({time.time()-t0:.1f}s)")

    n_missing = count_missing(cfg.RESULTS_SVM_DIR, seeds, data_sizes, n_outer)
    print(f"  SVM done. Still missing: {n_missing}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct per-sample SPEDIA predictions")
    parser.add_argument("--models", nargs="+", default=["nb", "svm"],
                        choices=["nb", "svm"],
                        help="Which models to reconstruct (default: nb svm)")
    args = parser.parse_args()

    print("=" * 60)
    print("SPEDIA Prediction Reconstruction")
    print(f"Models: {args.models}")
    print(f"Seeds:  {cfg.SEEDS}")
    print(f"Sizes:  {cfg.DATA_SIZES}")
    print("=" * 60)

    print("\nLoading SPEDIA features...")
    X_all, y_all = prepare_spedia_embeddings(cfg.SPEDIA_FEATURES_PATH)
    print(f"  {X_all.shape}  threat={y_all.sum()}  normal={(y_all==0).sum()}")

    seeds     = cfg.SEEDS
    data_sizes = cfg.DATA_SIZES
    n_outer   = cfg.N_OUTER_FOLDS

    if "nb" in args.models:
        reconstruct_nb(X_all, y_all, seeds, data_sizes, n_outer)

    if "svm" in args.models:
        reconstruct_svm(X_all, y_all, seeds, data_sizes, n_outer)

    print("\n" + "=" * 60)
    print("NB  predictions:", os.path.join(cfg.RESULTS_NB_DIR,  "predictions"))
    print("SVM predictions:", os.path.join(cfg.RESULTS_SVM_DIR, "predictions"))
    print("\nFor BiLSTM + Quantum, run:")
    print("  conda run -n qnlpv2 python3 spedia/main_singh.py  --force-preds")
    print("  conda run -n qnlpv2 python3 spedia/main_quantum.py --force-preds")
    print("=" * 60)


if __name__ == "__main__":
    main()
