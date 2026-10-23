import os
import sys
import random
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

_ALL_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR     = os.path.dirname(_ALL_DATA_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _ALL_DATA_DIR)

from src import config
from src.data_process import load_behavioral_features
from src.data_process import use_tuning_pool
use_tuning_pool()
from src.optuna_autowrite import autowrite_printed_config; import os as _os; autowrite_printed_config(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src', 'config.py'))

try:
    import optuna
except ImportError:
    os.system(f"{sys.executable} -m pip install optuna"); import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED       = 42
TUNING_N   = 3000
N_CV_FOLDS = 3
N_TRIALS   = 20
INPUT_DIM  = 16


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_tuning_pool():
    X, y = load_behavioral_features()
    np.random.seed(SEED)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    n_pos = max(1, int(TUNING_N * y.mean())); n_neg = TUNING_N - n_pos
    sel = np.concatenate([
        np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
    ])
    np.random.shuffle(sel)
    X_tune, y_tune = X[sel], y[sel]
    pos_w = float((y_tune == 0).sum() / max((y_tune == 1).sum(), 1))
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})  pos_weight={pos_w:.2f}")
    return X_tune, y_tune, pos_w


def build_knn_adj(X_train, X_test, k):
    n_tr = len(X_train); n_te = len(X_test); n = n_tr + n_te
    X_all = np.vstack([X_train, X_test])
    norms = np.linalg.norm(X_all, axis=1, keepdims=True) + 1e-10
    X_norm = X_all / norms
    nn_m = NearestNeighbors(n_neighbors=min(k + 1, n_tr), metric="cosine",
                             algorithm="auto", n_jobs=-1)
    nn_m.fit(X_norm[:n_tr])
    rows, cols = [], []
    _, idx_tt = nn_m.kneighbors(X_norm[:n_tr])
    for i, nb in enumerate(idx_tt):
        for j in nb:
            if j != i: rows += [i, j]; cols += [j, i]
    _, idx_te = nn_m.kneighbors(X_norm[n_tr:])
    for it, nb in enumerate(idx_te):
        ig = n_tr + it
        for j in nb: rows += [ig, j]; cols += [j, ig]
    for i in range(n): rows.append(i); cols.append(i)
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    d = np.asarray(A.sum(1)).flatten(); d_inv = np.where(d > 0, 1.0/np.sqrt(d), 0.0)
    D = sp.diags(d_inv); A_n = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([A_n.row, A_n.col]), dtype=torch.long)
    val = torch.tensor(A_n.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n))


class GCNLayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.lin = nn.Linear(in_d, out_d)
    def forward(self, A, H):
        return self.lin(torch.sparse.mm(A, H))


class ResHybnet(nn.Module):
    def __init__(self, input_dim, gcn_hidden, gcn_layers, gcn_dropout,
                 cnn_channels, cnn_kernel, mlp_hidden, dropout):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_drop   = nn.Dropout(gcn_dropout)
        in_d = input_dim
        for _ in range(gcn_layers):
            self.gcn_layers.append(GCNLayer(in_d, gcn_hidden)); in_d = gcn_hidden
        gcn_out = gcn_hidden

        effective_kernel = min(cnn_kernel, input_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=effective_kernel,
                      padding=effective_kernel // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        cnn_out = cnn_channels

        fused = gcn_out + cnn_out
        self.fc1     = nn.Linear(fused, mlp_hidden)
        self.fc_res  = nn.Linear(fused, mlp_hidden)
        self.bn      = nn.LayerNorm(mlp_hidden)
        self.drop    = nn.Dropout(dropout)
        self.out     = nn.Linear(mlp_hidden, 1)

    def forward(self, A, X):
        h = X
        for layer in self.gcn_layers:
            h = torch.relu(layer(A, h)); h = self.gcn_drop(h)
        gcn_emb = h

        cnn_in  = X.unsqueeze(1)
        cnn_emb = self.cnn(cnn_in).squeeze(-1)

        fused = torch.cat([gcn_emb, cnn_emb], dim=-1)
        out   = torch.relu(self.bn(self.fc1(fused) + self.fc_res(fused)))
        out   = self.drop(out)
        return self.out(out).squeeze(-1)


def main():
    print("=" * 60)
    print("OPTUNA ResHybnet (GCN+CNN) TUNING - REAL CLASS DISTRIBUTION")
    print(f"   Tuning pool: {TUNING_N} samples (~{int(TUNING_N*0.0468)} threat)")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print("=" * 60)

    X_tune, y_tune, pos_weight_val = get_tuning_pool()
    device = config.DEVICE

    def objective(trial):
        k            = trial.suggest_int("k_neighbors", 3, 10)
        gcn_hidden   = trial.suggest_categorical("gcn_hidden", [16, 32, 64])
        gcn_layers   = trial.suggest_int("gcn_layers", 1, 3)
        gcn_dropout  = trial.suggest_float("gcn_dropout", 0.1, 0.4)
        cnn_channels = trial.suggest_categorical("cnn_channels", [8, 16, 32])
        cnn_kernel   = trial.suggest_categorical("cnn_kernel", [3, 5])
        mlp_hidden   = trial.suggest_categorical("mlp_hidden", [16, 32, 64])
        dropout      = trial.suggest_float("dropout", 0.1, 0.4)
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        epochs       = trial.suggest_categorical("epochs", [50, 100])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        pos_w   = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        skf     = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s     = []

        for tr_idx, vl_idx in skf.split(X_tune, y_tune):
            set_all_seeds(SEED)
            X_tr, X_vl = X_tune[tr_idx], X_tune[vl_idx]
            y_tr, y_vl = y_tune[tr_idx], y_tune[vl_idx]
            n_tr = len(X_tr)

            A = build_knn_adj(X_tr, X_vl, k).to(device)
            X_all = torch.tensor(np.vstack([X_tr, X_vl]),
                                  dtype=torch.float32).to(device)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)

            model = ResHybnet(INPUT_DIM, gcn_hidden, gcn_layers, gcn_dropout,
                               cnn_channels, cnn_kernel, mlp_hidden,
                               dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                           weight_decay=weight_decay)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = loss_fn(model(A, X_all)[:n_tr], y_tr_t)
                loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(A, X_all)[n_tr:]).cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            f1s.append(f1_score(y_vl, preds, average="macro", zero_division=0))

        dummy = ResHybnet(INPUT_DIM, gcn_hidden, gcn_layers, gcn_dropout,
                           cnn_channels, cnn_kernel, mlp_hidden, dropout)
        trial.set_user_attr("n_params", sum(p.numel() for p in dummy.parameters()))
        trial.set_user_attr("f1_std", float(np.std(f1s)))
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize",
                                 study_name="reshybnet_all_tuning",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial; p = best.params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"   n_params={best.user_attrs.get('n_params','?')}")
    print(f"\nUpdate in all_data/src/config.py:")
    print(f"{'-'*60}")
    print(f"RESHYBNET_K_NEIGHBORS  = {p['k_neighbors']}")
    print(f"RESHYBNET_GCN_HIDDEN   = {p['gcn_hidden']}")
    print(f"RESHYBNET_GCN_LAYERS   = {p['gcn_layers']}")
    print(f"RESHYBNET_GCN_DROPOUT  = {p['gcn_dropout']:.6f}")
    print(f"RESHYBNET_CNN_CHANNELS = {p['cnn_channels']}")
    print(f"RESHYBNET_CNN_KERNEL   = {p['cnn_kernel']}")
    print(f"RESHYBNET_MLP_HIDDEN   = {p['mlp_hidden']}")
    print(f"RESHYBNET_DROPOUT      = {p['dropout']:.6f}")
    print(f"RESHYBNET_LR           = {p['lr']:.6f}")
    print(f"RESHYBNET_EPOCHS       = {p['epochs']}")
    print(f"RESHYBNET_WEIGHT_DECAY = {p['weight_decay']:.6f}")


if __name__ == "__main__":
    main()
