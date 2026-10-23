import torch
from torch import nn


class BehaviorTransformer(nn.Module):
    def __init__(self, input_dim: int = 16, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.2,
                 dim_feedforward: int = 64):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.input_proj = nn.Linear(1, d_model)

        self.pos_emb = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout  = nn.Dropout(dropout)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.input_proj(x) + self.pos_emb
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)
