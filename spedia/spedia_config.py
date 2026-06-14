"""
SPEDIA Pipeline Configuration
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


SPEDIA_DIR          = os.path.dirname(os.path.abspath(__file__))
SPEDIA_FEATURES_PATH = os.path.join(BASE_DIR, "data", "spedia_behavioral_features.pkl")

RESULTS_QUANTUM_DIR  = os.path.join(SPEDIA_DIR, "results_quantum")
RESULTS_SINGH_DIR    = os.path.join(SPEDIA_DIR, "results_singh")
RESULTS_NB_DIR       = os.path.join(SPEDIA_DIR, "results_nb")
RESULTS_SVM_DIR      = os.path.join(SPEDIA_DIR, "results_svm")

CHECKPOINT_QUANTUM_DIR = os.path.join(SPEDIA_DIR, "checkpoints_quantum")
CHECKPOINT_SINGH_DIR   = os.path.join(SPEDIA_DIR, "checkpoints_singh")

PAD_DIM   = 16   
INPUT_DIM = PAD_DIM

DATA_SIZES = [50, 100, 250, 500]
SEEDS      = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


N_OUTER_FOLDS      = 5
N_INNER_FOLDS      = 5
OPTUNA_INNER_TRIALS = 15

THREAT_RATIO = 0.4

# Hybrid VQC Hyperparameters 
N_QUBITS   = 7
N_LAYERS   = 2
NN_HIDDEN  = 8
NN_DEPTH   = 2
PRE_HIDDEN = 128
OPTUNA_INNER_TRIALS_QUANTUM = 10

# Naive Bayes Hyperparameter 
VAR_SMOOTHING = 2.869838e-02

# SVM Hyperparameters
SVM_C      = 2.869838e-02
SVM_KERNEL = "linear"
SVM_GAMMA  = "auto"

# For Singh et al.
BILSTM_HIDDEN  = 256
BILSTM_LAYERS  = 3
BILSTM_EPOCHS  = 110
BILSTM_LR      = 0.000112
BILSTM_BATCH   = 32
