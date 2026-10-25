import torch
from torch import nn

from .features import N_FEATURES


class IBDNet(nn.Module):
    """Small MLP: predicts P(clinical remission at week 52 | baseline, biologic)."""

    def __init__(self, n_features=N_FEATURES, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
