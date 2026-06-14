"""
BiLSTM Classifier — Singh et al. (J. Ambient Intell. Humanized Comput., 2023)
Adapted for 16-dim PCA behavioral features.
"""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out, _ = self.bilstm(x)         
        last = out[:, -1, :]             
        return self.classifier(last).squeeze(1) 
