"""Individual branches for multi-branch CNN (Proposal Bab 2.6.2).

Three branch architectures:
  (a) PSD branch: 1D convolution on spectrum vector (n_leads * n_freqs,)
  (b) Wavelet branch: 2D convolution on scalogram (n_leads, n_scales, window_len)
  (c) HOS branch: 1D convolution or dense on bispectrum features (n_leads * n_bins²,)

All branches output a flattened vector for concatenation in fusion layer.
"""
import torch
import torch.nn as nn


class PSDBlock(nn.Module):
    """1D CNN for PSD spectral features."""

    def __init__(self, input_size: int = 384, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.output_size = hidden * 16

    def forward(self, x):
        # x shape: (batch, input_size)
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        x = self.net(x)
        return x.view(x.size(0), -1)  # (batch, output_size)


class WaveletBlock(nn.Module):
    """2D CNN for wavelet scalogram features."""

    def __init__(self, n_leads: int = 12, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_leads, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, hidden, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.output_size = hidden * 4 * 4

    def forward(self, x):
        # x shape: (batch, n_leads, n_scales, window_len)
        x = self.net(x)
        return x.view(x.size(0), -1)  # (batch, output_size)


class HOSBlock(nn.Module):
    """1D/Dense branch for HOS bispectrum + moments features."""

    def __init__(self, input_size: int = 792, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.output_size = hidden * 16

    def forward(self, x):
        # x shape: (batch, input_size)
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        x = self.net(x)
        return x.view(x.size(0), -1)  # (batch, output_size)
