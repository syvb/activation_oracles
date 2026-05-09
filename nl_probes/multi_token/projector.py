"""Trainable modules for multi-token activation injection.

Two pieces:
  - MultiTokenProjector: linear map W: R^d -> R^(K x d). At init, W_1 = I and the
    other slots are small N(0, std) so K=1 with std=0 reproduces the baseline.
  - InjectionAdapter: residual MLP applied to the placeholder positions
    immediately after the layer-1 injection. The last linear is zero-initialized
    so the adapter is the identity at start.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTokenProjector(nn.Module):
    def __init__(self, d_model: int, k: int, init_std: float = 0.02):
        super().__init__()
        self.d_model = d_model
        self.k = k
        # nn.Linear(d, K*d) with no bias; weight shape (K*d, d).
        # First d rows correspond to W_1, etc.
        self.linear = nn.Linear(d_model, k * d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.normal_(mean=0.0, std=init_std)
            # Set W_1 = I so K=1 reproduces the baseline exactly.
            self.linear.weight[:d_model] = torch.eye(d_model)

    def forward(self, a_BD: torch.Tensor) -> torch.Tensor:
        """a_BD: (B, d_model) -> (B, K, d_model)."""
        B, D = a_BD.shape
        assert D == self.d_model
        out = self.linear(a_BD)
        return out.view(B, self.k, self.d_model)


class InjectionAdapter(nn.Module):
    """Residual MLP applied at placeholder positions after injection.

    f(x) = x + W2 @ gelu(W1 @ x), with W2 zero-initialized (identity at init).
    """

    def __init__(self, d_model: int, hidden_mult: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(x)))
