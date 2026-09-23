"""Multi-branch CNN model with feature fusion (Proposal Bab 3.1.6, 2.6.2)."""
import torch
import torch.nn as nn

from pvc_localization.models.branches import PSDBlock, WaveletBlock, HOSBlock


class FusionCNN(nn.Module):
    """Multi-branch CNN that fuses different features.

    Architecture:
      Branch 1: PSD (1D conv)
      Branch 2: Wavelet (2D conv on scalogram)
      Branch 3: HOS (1D conv)
      Fusion: concatenate all branches → dense classifier
    """

    def __init__(
        self,
        feature_types: list[str],
        num_classes: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.feature_types = feature_types
        self.branches = nn.ModuleDict()
        branch_output_sizes = {}

        if "psd" in feature_types:
            self.branches["psd"] = PSDBlock(hidden=hidden_dim)
            branch_output_sizes["psd"] = self.branches["psd"].output_size

        if "wavelet" in feature_types:
            self.branches["wavelet"] = WaveletBlock(hidden=hidden_dim)
            branch_output_sizes["wavelet"] = self.branches["wavelet"].output_size

        if "hos" in feature_types:
            self.branches["hos"] = HOSBlock(hidden=hidden_dim)
            branch_output_sizes["hos"] = self.branches["hos"].output_size

        total_input = sum(branch_output_sizes.values())

        self.classifier = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: dict with keys matching feature_types, e.g.
                   {"psd": Tensor, "wavelet": Tensor, "hos": Tensor}

        Returns:
            logits: (batch_size, num_classes)
        """
        branch_outputs = []
        for feature_type in self.feature_types:
            output = self.branches[feature_type](batch[feature_type])
            branch_outputs.append(output)

        fused = torch.cat(branch_outputs, dim=1)
        logits = self.classifier(fused)
        return logits


class BaselineCNN(nn.Module):
    """Simple 1D CNN baseline (no multi-branch)."""

    def __init__(self, input_size: int = 384, num_classes: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(hidden_dim * 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, psd: torch.Tensor) -> torch.Tensor:
        x = psd.unsqueeze(1)  # (batch, 1, input_size)
        return self.net(x)
