"""
Data Processing Pipeline for ThesisP2
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from src import config


# Behavioral Features (RAIT-IIT-2025)
_behavioral_cache = None

def load_behavioral_features(path: str = None):
    global _behavioral_cache
    if _behavioral_cache is not None:
        return _behavioral_cache

    path = path or config.BEHAVIORAL_FEATURES_PATH
    print(f"Loading behavioral features: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    X = data["X_pca"].astype(np.float32)
    y = data["y"].astype(int)
    print(f"   Loaded: {X.shape} | insider={y.sum()} normal={(y==0).sum()}")
    _behavioral_cache = (X, y)
    return X, y


def prepare_cert_embeddings(max_size: int, seed: int, cache_dir: str = None):
    X, y = load_behavioral_features()
    texts = np.array([""] * len(y))
    return X, y, texts

def create_nested_folds(labels, n_outer: int, n_inner: int, seed: int):
    outer_skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    dummy_X = np.zeros((len(labels), 1))
    nested = []

    for outer_train_idx, outer_test_idx in outer_skf.split(dummy_X, labels):
        inner_folds = []
        if n_inner >= 2:
            inner_skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
            inner_labels = labels[outer_train_idx]
            inner_dummy = np.zeros((len(inner_labels), 1))
            for inner_train_rel, inner_val_rel in inner_skf.split(inner_dummy, inner_labels):
                inner_train_abs = outer_train_idx[inner_train_rel]
                inner_val_abs = outer_train_idx[inner_val_rel]
                inner_folds.append((inner_train_abs, inner_val_abs))

        nested.append({
            "outer_train": outer_train_idx,
            "outer_test": outer_test_idx,
            "inner_folds": inner_folds,
        })

    return nested

def create_nonoverlapping_stages(embeddings, labels, data_sizes, seed,
                                  texts=None, threat_ratio=0.4):
    total_needed = sum(data_sizes)
    np.random.seed(seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_threat_total = int(total_needed * threat_ratio)
    n_normal_total = total_needed - n_threat_total

    sel_pos = np.random.choice(pos_idx, min(n_threat_total, len(pos_idx)), replace=False)
    sel_neg = np.random.choice(neg_idx, min(n_normal_total, len(neg_idx)), replace=False)

    np.random.shuffle(sel_pos)
    np.random.shuffle(sel_neg)

    stages = {}
    pos_offset = 0
    neg_offset = 0

    for size in data_sizes:
        n_threat = int(size * threat_ratio)
        n_normal = size - n_threat

        stage_pos = sel_pos[pos_offset:pos_offset + n_threat]
        stage_neg = sel_neg[neg_offset:neg_offset + n_normal]
        pos_offset += n_threat
        neg_offset += n_normal

        stage_idx = np.concatenate([stage_pos, stage_neg])
        np.random.shuffle(stage_idx)

        if texts is not None:
            stages[size] = (embeddings[stage_idx], labels[stage_idx], texts[stage_idx])
        else:
            stages[size] = (embeddings[stage_idx], labels[stage_idx])

    return stages


def create_tuning_eval_split(embeddings, labels, tuning_size=100, seed=42,
                              texts=None, threat_ratio=0.4):
    np.random.seed(seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_tune_threat = int(tuning_size * threat_ratio)
    n_tune_normal = tuning_size - n_tune_threat

    tune_pos = pos_idx[:n_tune_threat]
    tune_neg = neg_idx[:n_tune_normal]
    tune_idx = np.concatenate([tune_pos, tune_neg])
    np.random.shuffle(tune_idx)

    eval_pos = pos_idx[n_tune_threat:]
    eval_neg = neg_idx[n_tune_normal:]
    eval_idx = np.concatenate([eval_pos, eval_neg])
    np.random.shuffle(eval_idx)

    if texts is not None:
        tuning = (embeddings[tune_idx], labels[tune_idx], texts[tune_idx])
        evaluation = (embeddings[eval_idx], labels[eval_idx], texts[eval_idx])
    else:
        tuning = (embeddings[tune_idx], labels[tune_idx])
        evaluation = (embeddings[eval_idx], labels[eval_idx])

    return tuning, evaluation
