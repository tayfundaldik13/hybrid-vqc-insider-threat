"""
Shared Configuration for ThesisP2 Pipeline
"""
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_SPEDIA = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEHAVIORAL_FEATURES_PATH = os.path.join(BASE_DIR, "data", "behavioral_features.pkl")
SPEDIA_FEATURES_PATH = os.path.join(BASE_DIR, "data", "spedia_behavioral_features.pkl")
CERT_CSV_PATH = os.path.join(BASE_DIR, "data", "cert_ALL_DATA_formatted.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_3")
RESULTS_DIR = os.path.join(BASE_DIR, "results_3")
CACHE_DIR = os.path.join(BASE_DIR, "cache_3")

PAD_DIM = 16          

DATA_SIZES = [50, 100, 250, 500]   
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

N_OUTER_FOLDS = 5

OPTUNA_INNER_TRIALS = 15  
