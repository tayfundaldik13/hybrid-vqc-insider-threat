import numpy as np
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


def build_knn_adj(X_train: np.ndarray, X_test: np.ndarray,
                  k: int) -> torch.Tensor:
    n_tr = len(X_train); n_te = len(X_test); n = n_tr + n_te
    X_all  = np.vstack([X_train, X_test])
    norms  = np.linalg.norm(X_all, axis=1, keepdims=True) + 1e-10
    X_norm = X_all / norms

    nn_model = NearestNeighbors(n_neighbors=min(k + 1, n_tr),
                                 metric="cosine", algorithm="auto", n_jobs=-1)
    nn_model.fit(X_norm[:n_tr])

    rows, cols = [], []
    _, idx_tt = nn_model.kneighbors(X_norm[:n_tr])
    for i, nb in enumerate(idx_tt):
        for j in nb:
            if j != i: rows += [i, j]; cols += [j, i]
    _, idx_te = nn_model.kneighbors(X_norm[n_tr:])
    for i_te, nb in enumerate(idx_te):
        ig = n_tr + i_te
        for j in nb: rows += [ig, j]; cols += [j, ig]
    for i in range(n): rows.append(i); cols.append(i)

    vals = np.ones(len(rows), dtype=np.float32)
    A    = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    deg       = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D         = sp.diags(d_inv_sqrt)
    A_norm    = (D @ A @ D).tocoo()

    indices = torch.tensor(np.vstack([A_norm.row, A_norm.col]), dtype=torch.long)
    values  = torch.tensor(A_norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n))


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.sparse.mm(A, H))


class ResHybnet(nn.Module):
    def __init__(self, input_dim: int = 16, gcn_hidden: int = 32,
                 gcn_layers: int = 2, gcn_dropout: float = 0.2,
                 cnn_channels: int = 16, cnn_kernel: int = 3,
                 mlp_hidden: int = 32, dropout: float = 0.3):
        super().__init__()

        self.gcn_layers = nn.ModuleList()
        self.gcn_drop   = nn.Dropout(gcn_dropout)
        in_d = input_dim
        for _ in range(gcn_layers):
            self.gcn_layers.append(GCNLayer(in_d, gcn_hidden))
            in_d = gcn_hidden

        eff_kernel = min(cnn_kernel, input_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=eff_kernel,
                      padding=eff_kernel // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        fused = gcn_hidden + cnn_channels
        self.fc1    = nn.Linear(fused, mlp_hidden)
        self.fc_res = nn.Linear(fused, mlp_hidden)
        self.bn     = nn.LayerNorm(mlp_hidden)
        self.drop   = nn.Dropout(dropout)
        self.out    = nn.Linear(mlp_hidden, 1)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        h = X
        for layer in self.gcn_layers:
            h = torch.relu(layer(A, h))
            h = self.gcn_drop(h)

        cnn_emb = self.cnn(X.unsqueeze(1)).squeeze(-1)

        fused = torch.cat([h, cnn_emb], dim=-1)
        out   = torch.relu(self.bn(self.fc1(fused) + self.fc_res(fused)))
        out   = self.drop(out)
        return self.out(out).squeeze(-1)
