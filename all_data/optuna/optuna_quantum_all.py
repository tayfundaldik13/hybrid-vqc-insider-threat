import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import quantum_config as qconfig
ARCH_SEARCH = __import__("os").environ.get("QUANTUM_ARCH_SEARCH", "1") == "1"

from src.models.quantum_vqc import HybridQuantumModel
from src.data_process import load_behavioral_features
from src.data_process import use_tuning_pool
use_tuning_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config([_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'quantum_config.py'), _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'config.py')])

try:
    import optuna
except ImportError:
    os.system(f"{sys.executable} -m pip install optuna")
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED        = 42
TUNING_SIZE = 1000
N_CV_FOLDS  = 3
N_TRIALS    = 20


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tuning_pool():
    X, y = load_behavioral_features()
    np.random.seed(SEED)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    real_ratio = y.mean()
    n_pos = max(1, int(TUNING_SIZE * real_ratio))
    n_neg = TUNING_SIZE - n_pos

    sel = np.concatenate([
        np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
    ])
    np.random.shuffle(sel)

    tune_emb = X[sel]
    tune_lab = y[sel]
    pos_weight_val = (tune_lab == 0).sum() / max((tune_lab == 1).sum(), 1)

    print(f"   Tuning pool: {len(tune_lab)} samples  "
          f"(threat={int(tune_lab.sum())}, normal={int((tune_lab==0).sum())}, "
          f"ratio={tune_lab.mean():.4f})")
    print(f"   pos_weight for BCEWithLogitsLoss: {pos_weight_val:.2f}")
    return tune_emb, tune_lab, float(pos_weight_val)


def make_objective(tune_emb, tune_lab, pos_weight_val):

    def objective(trial):
        n_qubits   = trial.suggest_categorical("n_qubits",   [2, 4, 5, 6, 7, 8, 9]) if ARCH_SEARCH else qconfig.N_QUBITS
        n_layers   = trial.suggest_int("n_layers", 1, 5) if ARCH_SEARCH else qconfig.N_LAYERS
        lr         = trial.suggest_float("learning_rate", 1e-4, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        epochs     = trial.suggest_categorical("epochs", [20, 40, 60])
        nn_hidden  = trial.suggest_categorical("nn_hidden",  [8, 16, 32]) if ARCH_SEARCH else qconfig.NN_HIDDEN
        nn_depth   = trial.suggest_categorical("nn_depth",   [1, 2]) if ARCH_SEARCH else qconfig.NN_DEPTH
        pre_hidden = trial.suggest_categorical("pre_hidden", [0, 32, 64, 128]) if ARCH_SEARCH else qconfig.PRE_HIDDEN
        skf  = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s  = []
        pos_w = torch.tensor([pos_weight_val], dtype=torch.float32)

        for tr_idx, vl_idx in skf.split(tune_emb, tune_lab):
            set_all_seeds(SEED)

            X_tr = torch.tensor(tune_emb[tr_idx], dtype=torch.float32)
            X_vl = torch.tensor(tune_emb[vl_idx], dtype=torch.float32)
            y_tr = torch.tensor(tune_lab[tr_idx],  dtype=torch.float32)
            y_vl = tune_lab[vl_idx]

            eff_bs   = max(2, min(batch_size, len(X_tr) - 1))
            train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                                  batch_size=eff_bs, shuffle=True, drop_last=True)

            model = HybridQuantumModel(
                input_dim=qconfig.INPUT_DIM,
                n_qubits=n_qubits, n_layers=n_layers,
                nn_hidden=nn_hidden, nn_depth=nn_depth, pre_hidden=pre_hidden,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_w)

            model.train()
            for _ in range(epochs):
                for xb, yb in train_dl:
                    logits = model(xb)
                    loss   = loss_fn(logits, yb.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(X_vl)).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(float)

            f1s.append(f1_score(y_vl, preds, average="macro", zero_division=0))

        set_all_seeds(SEED)
        dummy = HybridQuantumModel(
            input_dim=qconfig.INPUT_DIM,
            n_qubits=n_qubits, n_layers=n_layers,
            nn_hidden=nn_hidden, nn_depth=nn_depth, pre_hidden=pre_hidden,
        )
        trial.set_user_attr("n_params", sum(p.numel() for p in dummy.parameters()))
        trial.set_user_attr("amp_dim",  2 ** n_qubits)
        trial.set_user_attr("f1_std",   float(np.std(f1s)))
        return float(np.mean(f1s))

    return objective


def main():
    print("=" * 70)
    print("OPTUNA QUANTUM ARCH SEARCH - REAL CLASS DISTRIBUTION")
    print(f"   Tuning pool: {TUNING_SIZE} samples at real ratio (~{int(TUNING_SIZE*0.0468)} threat)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print(f"   Imbalance handled via pos_weight in BCEWithLogitsLoss")
    print("=" * 70)

    tune_emb, tune_lab, pos_weight_val = get_tuning_pool()

    study = optuna.create_study(
        direction="maximize",
        study_name="quantum_all_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(make_objective(tune_emb, tune_lab, pos_weight_val),
                   n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    p    = best.params
    print(f"\n{'='*70}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"   n_params={best.user_attrs.get('n_params','?')}  "
          f"amp_dim={best.user_attrs.get('amp_dim','?')}")
    print(f"\nCopy into src/quantum_config.py:")
    print(f"{'-'*70}")
    print(f"N_QUBITS   = {p.get('n_qubits', qconfig.N_QUBITS)}")
    print(f"N_LAYERS   = {p.get('n_layers', qconfig.N_LAYERS)}")
    print(f"NN_HIDDEN  = {p.get('nn_hidden', qconfig.NN_HIDDEN)}")
    print(f"NN_DEPTH   = {p.get('nn_depth', qconfig.NN_DEPTH)}")
    print(f"PRE_HIDDEN = {p.get('pre_hidden', qconfig.PRE_HIDDEN)}")
    print(f"LEARNING_RATE = {p['learning_rate']:.6f}")
    print(f"BATCH_SIZE = {p['batch_size']}")
    print(f"EPOCHS = {p['epochs']}")
    print(f"# (lr/batch/epochs re-tuned per-fold inside main_quantum_all.py)")

    print(f"\nTOP 5 TRIALS:")
    trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        print(f"  #{i+1} MacroF1={t.value:.4f}+/-{t.user_attrs.get('f1_std',0):.4f} | "
              f"q={t.params.get('n_qubits')} L={t.params.get('n_layers')} "
              f"lr={t.params.get('learning_rate',0):.5f} "
              f"bs={t.params.get('batch_size')} ep={t.params.get('epochs')} "
              f"nn_h={t.params.get('nn_hidden')} pre_h={t.params.get('pre_hidden')} "
              f"params={t.user_attrs.get('n_params','?')}")


if __name__ == "__main__":
    main()
