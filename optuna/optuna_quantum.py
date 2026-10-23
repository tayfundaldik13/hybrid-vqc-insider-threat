import os
if os.environ.get("QUANTUM_OPTUNA_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src import quantum_config as qconfig
ARCH_SEARCH = __import__("os").environ.get("QUANTUM_ARCH_SEARCH", "1") == "1"

from src.models.quantum_vqc import HybridQuantumModel
from src.data_process import prepare_cert_embeddings, create_tuning_eval_split
from src.data_process import GroupAwareStratifiedKFold as StratifiedKFold, use_full_pool
use_full_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'quantum_config.py'))


try:
    import optuna
except ImportError:
    print("Installing optuna...")
    os.system(f"{sys.executable} -m pip install optuna")
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS = [42, 43, 44]
TOTAL_POOL_SIZE = 990
TUNING_SIZE = 100
N_CV_FOLDS = 3
N_TRIALS = 30


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tuning_pools():
    tuning_pools = {}
    for seed in SEEDS:
        full_emb, full_lab = prepare_cert_embeddings(
            max_size=TOTAL_POOL_SIZE, seed=seed
        )
        tuning, _ = create_tuning_eval_split(
            full_emb, full_lab, tuning_size=TUNING_SIZE, seed=seed
        )
        tune_emb, tune_lab = tuning
        tuning_pools[seed] = (tune_emb, tune_lab)
        print(f"   Seed {seed}: tuning={len(tune_emb)} "
              f"(threat={int(tune_lab.sum())}, normal={int((tune_lab == 0).sum())})")
    return tuning_pools


def _fit_fold(task):
    seed, Xtr, ytr, Xvl, yvl, hp = task
    torch.set_num_threads(1)
    set_all_seeds(seed)
    X_train = torch.tensor(Xtr, dtype=torch.float32)
    X_val   = torch.tensor(Xvl, dtype=torch.float32)
    y_train = torch.tensor(ytr, dtype=torch.float32)
    y_val   = torch.tensor(yvl, dtype=torch.float32)

    eff_bs = max(2, min(hp["batch_size"], len(X_train) - 1))
    dl = DataLoader(TensorDataset(X_train, y_train), batch_size=eff_bs, shuffle=True, drop_last=True)
    model = HybridQuantumModel(
        input_dim=qconfig.INPUT_DIM, n_qubits=hp["n_qubits"], n_layers=hp["n_layers"],
        nn_hidden=hp["nn_hidden"], nn_depth=hp["nn_depth"], pre_hidden=hp["pre_hidden"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
    _ytr = np.asarray(ytr)
    _pw = torch.tensor([float((_ytr == 0).sum() / max((_ytr == 1).sum(), 1))], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pw)
    model.train()
    for _ in range(hp["epochs"]):
        for xb, yb in dl:
            loss = loss_fn(model(xb), yb.unsqueeze(1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_val)).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(float)
    return f1_score(y_val.numpy().flatten(), preds, average="macro", zero_division=0)


_POOL = None


def _start_pool():
    global _POOL
    w = int(os.environ.get("QUANTUM_OPTUNA_WORKERS", str(min(9, os.cpu_count() or 1))))
    if w > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        _POOL = ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("fork"))
        print(f"   Parallel: {w} worker processes (one (seed, fold) fit each)")


def _run_tasks(tasks):
    if _POOL is None:
        return [_fit_fold(t) for t in tasks]
    return list(_POOL.map(_fit_fold, tasks))


def make_objective(tuning_pools):

    def objective(trial):
        n_qubits = trial.suggest_categorical("n_qubits", [2, 4, 5, 6, 7, 8, 9]) if ARCH_SEARCH else qconfig.N_QUBITS
        n_layers = trial.suggest_int("n_layers", 1, 5) if ARCH_SEARCH else qconfig.N_LAYERS
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [5, 8, 10, 16])
        epochs = trial.suggest_int("epochs", 20, 100)
        nn_hidden = trial.suggest_categorical("nn_hidden", [8, 16, 32]) if ARCH_SEARCH else qconfig.NN_HIDDEN
        nn_depth = trial.suggest_categorical("nn_depth", [1, 2]) if ARCH_SEARCH else qconfig.NN_DEPTH
        pre_hidden = trial.suggest_categorical("pre_hidden", [0, 32, 64, 128]) if ARCH_SEARCH else qconfig.PRE_HIDDEN
        hp = dict(n_qubits=n_qubits, n_layers=n_layers, lr=lr, batch_size=batch_size, epochs=epochs,
                  nn_hidden=nn_hidden, nn_depth=nn_depth, pre_hidden=pre_hidden)
        tasks = []
        for seed in SEEDS:
            tune_emb, tune_lab = tuning_pools[seed]
            skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
            for train_idx, val_idx in skf.split(tune_emb, tune_lab):
                tasks.append((seed, tune_emb[train_idx], tune_lab[train_idx],
                              tune_emb[val_idx], tune_lab[val_idx], hp))
        all_f1s = _run_tasks(tasks)

        mean_f1 = np.mean(all_f1s)

        set_all_seeds(42)
        dummy = HybridQuantumModel(
            input_dim=qconfig.INPUT_DIM,
            n_qubits=n_qubits, n_layers=n_layers,
            nn_hidden=nn_hidden, nn_depth=nn_depth,
            pre_hidden=pre_hidden,
        )
        n_params = sum(p.numel() for p in dummy.parameters())
        trial.set_user_attr("n_params", n_params)
        trial.set_user_attr("amp_dim", 2 ** n_qubits)
        trial.set_user_attr("f1_std", float(np.std(all_f1s)))

        return mean_f1

    return objective


def main():
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING - Hybrid Quantum VQC")
    print(f"   Architecture: Pre-Quantum -> AmplitudeEmbedding -> StronglyEntanglingLayers -> Post-Quantum")
    print(f"   Tuning pool: {TUNING_SIZE} samples (ISOLATED - never seen during evaluation)")
    print(f"   Seeds: {SEEDS} x {N_CV_FOLDS}-fold CV = {len(SEEDS) * N_CV_FOLDS} evals/trial")
    print(f"   Trials: {N_TRIALS}")
    print(f"   Qubit search: [2, 4, 5, 6, 7, 8, 9]")
    print(f"   Pre-hidden search: [0, 32, 64, 128]")
    print(f"   Layer search: [1, 2, 3, 4, 5]")
    print("=" * 70)

    print(f"\nPreparing tuning pools ({TUNING_SIZE} samples x {len(SEEDS)} seeds)...")
    tuning_pools = get_tuning_pools()

    print(f"\n{'=' * 70}")
    print(f"Running Optuna ({N_TRIALS} trials)...")
    print(f"{'=' * 70}")

    _start_pool()
    study = optuna.create_study(
        direction="maximize",
        study_name="quantum_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective_fn = make_objective(tuning_pools)
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'=' * 70}")
    print(f"BEST TRIAL")
    print(f"{'=' * 70}")
    best = study.best_trial
    print(f"   Macro-F1 (mean across {len(SEEDS)}seeds x {N_CV_FOLDS}folds): {best.value:.4f} "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"   Trainable params: {best.user_attrs.get('n_params', '?')}")
    print(f"   Amplitude dim: {best.user_attrs.get('amp_dim', '?')}")

    print(f"\nCopy-paste these into src/quantum_config.py:")
    print(f"{'-' * 70}")
    p = best.params
    print(f"N_QUBITS = {p.get('n_qubits', qconfig.N_QUBITS)}")
    print(f"N_LAYERS = {p.get('n_layers', qconfig.N_LAYERS)}")
    print(f"LEARNING_RATE = {p['learning_rate']:.6f}")
    print(f"BATCH_SIZE = {p['batch_size']}")
    print(f"EPOCHS = {p['epochs']}")
    print(f"NN_HIDDEN = {p.get('nn_hidden', qconfig.NN_HIDDEN)}")
    print(f"NN_DEPTH = {p.get('nn_depth', qconfig.NN_DEPTH)}")
    print(f"PRE_HIDDEN = {p.get('pre_hidden', qconfig.PRE_HIDDEN)}")

    print(f"\n{'-' * 70}")
    print("TOP 5 TRIALS:")
    print(f"{'-' * 70}")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(trials[:5]):
        n_p = t.user_attrs.get("n_params", "?")
        std = t.user_attrs.get("f1_std", 0)
        print(f"  #{i+1} Macro-F1={t.value:.4f}+/-{std:.4f} | q={t.params.get('n_qubits')} "
              f"L={t.params.get('n_layers')} "
              f"lr={t.params.get('learning_rate', 0):.5f} "
              f"bs={t.params.get('batch_size')} "
              f"ep={t.params.get('epochs')} "
              f"nn_h={t.params.get('nn_hidden')} "
              f"nn_d={t.params.get('nn_depth')} "
              f"pre_h={t.params.get('pre_hidden')} "
              f"params={n_p}")


if __name__ == "__main__":
    main()
