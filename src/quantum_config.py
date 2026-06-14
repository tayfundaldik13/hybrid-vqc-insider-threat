"""
Quantum-specific Configuration for VQC
"""
from src.config import (
    SEEDS, DATA_SIZES, N_OUTER_FOLDS, N_INNER_FOLDS,
    CERT_CSV_PATH, ENRON_CSV_PATH, BASE_DIR, PAD_DIM
)
import os


N_QUBITS = 9              # Optuna optimized
N_LAYERS = 2              # Optuna optimized
INPUT_DIM = PAD_DIM      

NN_HIDDEN = 8             # Optuna optimized
NN_DEPTH = 1              # Optuna optimized
PRE_HIDDEN = 0            # Optuna optimized: single Linear(16→2^9=512)

LEARNING_RATE = 0.010438  # Optuna optimized
EPOCHS = 40               # Optuna optimized
BATCH_SIZE = 5            # Optuna optimized

OPTUNA_INNER_TRIALS = 10  

QUANTUM_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_3", "quantum")
QUANTUM_RESULTS_DIR = os.path.join(BASE_DIR, "results_3", "quantum")
