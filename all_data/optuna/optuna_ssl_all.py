import os
import sys
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

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
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def get_tuning_pool():
    X, y = load_behavioral_features()
    np.random.seed(SEED)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    n_pos = max(1, int(TUNING_N * y.mean())); n_neg = TUNING_N - n_pos
    sel = np.concatenate([
        np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)])
    np.random.shuffle(sel)
    X_tune, y_tune = X[sel], y[sel]
    pos_w = float((y_tune == 0).sum() / max((y_tune == 1).sum(), 1))
    print(f"   Tuning pool: {len(y_tune)} samples  "
          f"(threat={int(y_tune.sum())}, normal={int((y_tune==0).sum())}, "
          f"ratio={y_tune.mean():.4f})  pos_weight={pos_w:.2f}")
    return X_tune, y_tune, pos_w


def corrupt_features(X, corruption_rate, marginal_samples):
    mask = torch.rand_like(X) < corruption_rate
    n    = X.shape[0]
    rand_idx = torch.randint(0, len(marginal_samples), (n,))
    X_corrupt = marginal_samples[rand_idx].to(X.device)
    return torch.where(mask, X_corrupt, X)


def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n  = z1.shape[0]
    z  = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, -1e9)
    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels], dim=0)
    return F.cross_entropy(sim, labels)


class SCARFEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, n_layers, dropout):
        super().__init__()
        layers = []
        in_d   = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        self.encoder   = nn.Sequential(*layers)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        return self.projector(self.encode(x))


class SSLClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(self.encoder.encode(x)).squeeze(-1)


def main():
    print("=" * 60)
    print("OPTUNA Self-Supervised (SCARF) TUNING - REAL CLASS DISTRIBUTION")
    print(f"   Tuning pool: {TUNING_N} samples (~{int(TUNING_N*0.0468)} threat)")
    print(f"   Phase 1: SCARF contrastive pretraining (unsupervised)")
    print(f"   Phase 2: Supervised fine-tuning with pos_weight")
    print(f"   CV: {N_CV_FOLDS} folds  |  Trials: {N_TRIALS}")
    print("=" * 60)

    X_tune, y_tune, pos_weight_val = get_tuning_pool()
    device = config.DEVICE

    def objective(trial):
        hidden_dim       = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        proj_dim         = trial.suggest_categorical("proj_dim", [16, 32])
        n_encoder_layers = trial.suggest_int("n_encoder_layers", 2, 4)
        dropout          = trial.suggest_float("dropout", 0.1, 0.4)
        corruption_rate  = trial.suggest_float("corruption_rate", 0.1, 0.6)
        temperature      = trial.suggest_float("temperature", 0.1, 1.0)
        pretrain_lr      = trial.suggest_float("pretrain_lr", 1e-4, 1e-2, log=True)
        pretrain_epochs  = trial.suggest_categorical("pretrain_epochs", [50, 100])
        finetune_lr      = trial.suggest_float("finetune_lr", 1e-4, 1e-2, log=True)
        finetune_epochs  = trial.suggest_categorical("finetune_epochs", [30, 50, 100])
        batch_size       = trial.suggest_categorical("batch_size", [32, 64])
        freeze_encoder   = trial.suggest_categorical("freeze_encoder", [True, False])

        pos_w   = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        skf     = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        f1s     = []

        for tr_idx, vl_idx in skf.split(X_tune, y_tune):
            set_all_seeds(SEED)
            X_tr_np = X_tune[tr_idx]; X_vl_np = X_tune[vl_idx]
            y_tr_np = y_tune[tr_idx]; y_vl_np = y_tune[vl_idx]

            X_tr = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
            X_vl = torch.tensor(X_vl_np, dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_tr_np, dtype=torch.float32).to(device)

            marginals = X_tr.cpu()

            encoder   = SCARFEncoder(INPUT_DIM, hidden_dim, proj_dim,
                                      n_encoder_layers, dropout).to(device)
            opt_pre   = torch.optim.Adam(encoder.parameters(), lr=pretrain_lr)
            eff_bs    = max(4, min(batch_size, len(X_tr) - 1))
            dl_pre    = DataLoader(TensorDataset(X_tr),
                                    batch_size=eff_bs, shuffle=True, drop_last=True)

            encoder.train()
            for _ in range(pretrain_epochs):
                for (xb,) in dl_pre:
                    x_corrupt = corrupt_features(xb, corruption_rate, marginals).to(device)
                    z1 = encoder.project(xb)
                    z2 = encoder.project(x_corrupt)
                    loss = nt_xent_loss(z1, z2, temperature)
                    opt_pre.zero_grad(); loss.backward(); opt_pre.step()

            if freeze_encoder:
                for p in encoder.encoder.parameters(): p.requires_grad = False

            clf       = SSLClassifier(encoder, hidden_dim).to(device)
            opt_ft    = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, clf.parameters()),
                lr=finetune_lr)
            dl_ft     = DataLoader(TensorDataset(X_tr, y_tr),
                                    batch_size=eff_bs, shuffle=True, drop_last=True)

            clf.train()
            for _ in range(finetune_epochs):
                for xb, yb in dl_ft:
                    opt_ft.zero_grad()
                    loss = loss_fn(clf(xb), yb)
                    loss.backward(); opt_ft.step()

            clf.eval()
            with torch.no_grad():
                probs = torch.sigmoid(clf(X_vl)).cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            f1s.append(f1_score(y_vl_np, preds, average="macro", zero_division=0))

        dummy_enc = SCARFEncoder(INPUT_DIM, hidden_dim, proj_dim, n_encoder_layers, dropout)
        dummy_clf = SSLClassifier(dummy_enc, hidden_dim)
        trial.set_user_attr("n_params", sum(p.numel() for p in dummy_clf.parameters()))
        trial.set_user_attr("f1_std",   float(np.std(f1s)))
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize",
                                 study_name="ssl_all_tuning",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial; p = best.params
    print(f"\n{'='*60}")
    print(f"BEST  -  Macro-F1={best.value:.4f}  "
          f"(+/-{best.user_attrs.get('f1_std', 0):.4f})")
    print(f"   n_params={best.user_attrs.get('n_params','?')}")
    print(f"\nUpdate in all_data/src/config.py:")
    print(f"{'-'*60}")
    print(f"SSL_HIDDEN_DIM       = {p['hidden_dim']}")
    print(f"SSL_PROJ_DIM         = {p['proj_dim']}")
    print(f"SSL_N_ENCODER_LAYERS = {p['n_encoder_layers']}")
    print(f"SSL_DROPOUT          = {p['dropout']:.6f}")
    print(f"SSL_CORRUPTION_RATE  = {p['corruption_rate']:.6f}")
    print(f"SSL_TEMPERATURE      = {p['temperature']:.6f}")
    print(f"SSL_PRETRAIN_LR      = {p['pretrain_lr']:.6f}")
    print(f"SSL_PRETRAIN_EPOCHS  = {p['pretrain_epochs']}")
    print(f"SSL_FINETUNE_LR      = {p['finetune_lr']:.6f}")
    print(f"SSL_FINETUNE_EPOCHS  = {p['finetune_epochs']}")
    print(f"SSL_BATCH_SIZE       = {p['batch_size']}")
    print(f"SSL_FREEZE_ENCODER   = {p['freeze_encoder']}")


if __name__ == "__main__":
    main()
