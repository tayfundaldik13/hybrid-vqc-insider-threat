"""
Shared Configuration for r5.2 Pipeline
"""
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEHAVIORAL_FEATURES_PATH = os.path.join(BASE_DIR, "data", "behavioral_features.pkl")

RESULTS_QUANTUM_DIR  = os.path.join(BASE_DIR, "results_quantum")
RESULTS_SINGH_DIR    = os.path.join(BASE_DIR, "results_singh")
RESULTS_NB_DIR       = os.path.join(BASE_DIR, "results_nb")
RESULTS_SVM_DIR      = os.path.join(BASE_DIR, "results_svm")

CHECKPOINT_QUANTUM_DIR = os.path.join(BASE_DIR, "checkpoints_quantum")
CHECKPOINT_SINGH_DIR   = os.path.join(BASE_DIR, "checkpoints_singh")

PAD_DIM   = 16   
INPUT_DIM = PAD_DIM
INPUT_SIZE = PAD_DIM

DATA_SIZES = [50, 100, 250, 500]
SEEDS      = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

N_OUTER_FOLDS       = 5
N_INNER_FOLDS       = 5
OPTUNA_INNER_TRIALS = 15


THREAT_RATIO = 0.4

# VQC Hyperparameters
N_QUBITS   = 9
N_LAYERS   = 4
NN_HIDDEN  = 8
NN_DEPTH   = 2
PRE_HIDDEN = 32
OPTUNA_INNER_TRIALS_QUANTUM = 10

# Naive Bayes Hyperparameter
VAR_SMOOTHING = 5.4831e-02

# SVM Hyperparameters
SVM_C      = 0.541441
SVM_KERNEL = "rbf"
SVM_GAMMA  = "scale"

# Singh BiLSTM Hyperparameters
BILSTM_HIDDEN  = 256
BILSTM_LAYERS  = 3
BILSTM_EPOCHS  = 140
BILSTM_LR      = 0.005961
BILSTM_BATCH   = 16
