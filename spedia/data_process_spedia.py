"""
SPEDIA Data Loading
"""
import pickle
import numpy as np
import warnings

_spedia_cache = None

def prepare_spedia_embeddings(path: str):
    global _spedia_cache
    if _spedia_cache is not None:
        return _spedia_cache

    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pickle.load(f)

    X = data["X_pca"].astype(np.float32)
    y = data["y"].astype(int)
    texts = np.array([""] * len(X))
    print(f"SPEDIA loaded: {X.shape} | threat={y.sum()} normal={(y==0).sum()}")
    _spedia_cache = (X, y, texts)
    return _spedia_cache

def create_nonoverlapping_stages(embeddings, labels, data_sizes, seed, texts=None, threat_ratio=0.4):
    total_needed = sum(data_sizes)
    rng = np.random.RandomState(seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_threat_total = int(total_needed * threat_ratio)
    n_normal_total = total_needed - n_threat_total

    sel_pos = rng.choice(pos_idx, min(n_threat_total, len(pos_idx)), replace=False)
    sel_neg = rng.choice(neg_idx, min(n_normal_total, len(neg_idx)), replace=False)

    rng.shuffle(sel_pos)
    rng.shuffle(sel_neg)

    stages = {}
    pos_offset = 0
    neg_offset = 0

    for size in data_sizes:
        n_threat = int(size * threat_ratio)
        n_normal = size - n_threat

        p_idx = sel_pos[pos_offset: pos_offset + n_threat]
        n_idx = sel_neg[neg_offset: neg_offset + n_normal]
        pos_offset += n_threat
        neg_offset += n_normal

        idx = np.concatenate([p_idx, n_idx])
        rng.shuffle(idx)

        t = texts[idx] if texts is not None else None
        stages[size] = (embeddings[idx], labels[idx], t)

    return stages

def create_tuning_eval_split(embeddings, labels, tuning_size=100, seed=42,
                              texts=None, threat_ratio=0.4):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_tune_threat = int(tuning_size * threat_ratio)
    n_tune_normal = tuning_size - n_tune_threat

    tune_idx = np.concatenate([pos_idx[:n_tune_threat], neg_idx[:n_tune_normal]])
    eval_idx = np.concatenate([pos_idx[n_tune_threat:], neg_idx[n_tune_normal:]])
    rng.shuffle(tune_idx)
    rng.shuffle(eval_idx)

    if texts is not None:
        return (embeddings[tune_idx], labels[tune_idx], texts[tune_idx]), \
               (embeddings[eval_idx], labels[eval_idx], texts[eval_idx])
    return (embeddings[tune_idx], labels[tune_idx]), \
           (embeddings[eval_idx], labels[eval_idx])

def create_nested_folds(labels, n_outer, n_inner=1, seed=42):
    from sklearn.model_selection import StratifiedKFold
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    folds = []
    for outer_train, outer_test in outer_cv.split(np.zeros(len(labels)), labels):
        folds.append({"outer_train": outer_train, "outer_test": outer_test})
    return folds
