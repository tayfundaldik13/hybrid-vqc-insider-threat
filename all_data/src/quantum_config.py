import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR  = os.path.dirname(os.path.dirname(_THIS_DIR))

BEHAVIORAL_FEATURES_PATH = os.path.join(
    _BASE_DIR, "data", "behavioral_features.pkl")

INPUT_DIM = 16

N_QUBITS   = 9
N_LAYERS   = 2
NN_HIDDEN  = 8
NN_DEPTH   = 1
PRE_HIDDEN = 0

OPTUNA_INNER_TRIALS = 10
