"""Individual branches for multi-branch CNN (Proposal Bab 2.6.2).

Three branch architectures:
  (a) PSD branch: 1D convolution on spectrum vector (n_leads * n_freqs,)
  (b) Wavelet branch: 2D convolution on scalogram (n_leads, n_scales, n_time)
  (c) HOS branch: 1D convolution on bispectrum features (n_leads * n_bins² + moments,)

Each branch has two conv layers with hidden // 2 and hidden filters, and outputs a
flattened vector for concatenation in the fusion layer.
"""
import torch
import torch.nn as nn


class _Conv1DBranch(nn.Module):
    def __init__(self, input_size: int, hidden: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.input_norm = nn.BatchNorm1d(input_size)
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden // 2, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden // 2, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.output_size = hidden * 16

    def forward(self, x):
        # x shape: (batch, input_size)
        x = self.net(self.input_norm(x).unsqueeze(1))
        return x.view(x.size(0), -1)


class PSDBlock(_Conv1DBranch):
    """1D CNN for PSD spectral features."""

    def __init__(self, input_size: int = 384, hidden: int = 64, kernel_size: int = 5):
        super().__init__(input_size, hidden, kernel_size)


class HOSBlock(_Conv1DBranch):
    """1D CNN for HOS bispectrum + moments features."""

    def __init__(self, input_size: int = 792, hidden: int = 64, kernel_size: int = 5):
        super().__init__(input_size, hidden, kernel_size)


class WaveletBlock(nn.Module):
    """2D CNN for wavelet scalogram features."""

    def __init__(self, n_leads: int = 12, hidden: int = 64, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.input_norm = nn.BatchNorm2d(n_leads)
        self.net = nn.Sequential(
            nn.Conv2d(n_leads, hidden // 2, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(hidden // 2, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.output_size = hidden * 4 * 4

    def forward(self, x):
        # x shape: (batch, n_leads, n_scales, n_time)
        x = self.net(self.input_norm(x))
        return x.view(x.size(0), -1)
