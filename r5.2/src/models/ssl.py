import torch
from torch import nn
import torch.nn.functional as F


def corrupt_features(X: torch.Tensor, corruption_rate: float,
                     marginal_samples: torch.Tensor) -> torch.Tensor:
    mask     = torch.rand_like(X) < corruption_rate
    rand_idx = torch.randint(0, len(marginal_samples), (X.shape[0],))
    X_corrupt = marginal_samples[rand_idx].to(X.device)
    return torch.where(mask, X_corrupt, X)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
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
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64,
                 proj_dim: int = 32, n_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []; in_d = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_d = hidden_dim
        self.encoder   = nn.Sequential(*layers)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encode(x))


class SSLClassifier(nn.Module):
    def __init__(self, encoder: SCARFEncoder, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.encode(x)).squeeze(-1)
