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
        n_filters: int = 64,
        kernel_size: int = None,
        dropout: float = 0.3,
    ):
        """kernel_size=None keeps each branch's default (5 for 1D, 3 for 2D)."""
        super().__init__()
        self.feature_types = feature_types
        self.branches = nn.ModuleDict()
        kw = {"hidden": n_filters}
        if kernel_size is not None:
            kw["kernel_size"] = kernel_size

        if "psd" in feature_types:
            self.branches["psd"] = PSDBlock(**kw)
        if "wavelet" in feature_types:
            self.branches["wavelet"] = WaveletBlock(**kw)
        if "hos" in feature_types:
            self.branches["hos"] = HOSBlock(**kw)

        total_input = sum(branch.output_size for branch in self.branches.values())

        self.classifier = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    """1D CNN on the raw 12-lead beat, without hand-crafted features.

    The defaults (n_filters=64, kernel_size=None, dropout=0.3) give the original fixed architecture:
    32/64/64 filters with kernels 7/7/5. They are exposed so the baseline can be tuned with the same
    budget as the feature models (reviewer comment 6): kernel_size k gives kernels k/k/max(k-2, 3).
    """

    def __init__(self, n_leads: int = 12, num_classes: int = 2, hidden_dim: int = 64,
                 n_filters: int = 64, kernel_size: int = None, dropout: float = 0.3):
        super().__init__()
        k1 = 7 if kernel_size is None else kernel_size
        k3 = 5 if kernel_size is None else max(kernel_size - 2, 3)
        f = n_filters
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, f // 2, kernel_size=k1, padding=k1 // 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(f // 2, f, kernel_size=k1, padding=k1 // 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(f, f, kernel_size=k3, padding=k3 // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(f * 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.net(raw)  # raw: (batch, n_leads, window_len)
