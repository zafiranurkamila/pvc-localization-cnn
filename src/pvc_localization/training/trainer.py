"""K-Fold CV training loop for CNN models."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from pvc_localization import config
from pvc_localization.data.dataset import PVCBeatsDataset
from pvc_localization.models.fusion import FusionCNN, BaselineCNN
from pvc_localization.evaluation.metrics import compute_metrics

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


class CVTrainer:
    """Patient-level K-fold CV, then a final fit on all training patients evaluated on the test set."""

    def __init__(self, feature_types: list[str], num_folds: int = 5, device: str = None):
        self.feature_types = feature_types
        self.num_folds = num_folds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(
        self,
        train_ids: list[int],
        test_ids: list[int],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> tuple[list[dict], dict]:
        """Returns (fold_results, test_metrics)."""
        train_set = PVCBeatsDataset(train_ids, self.feature_types, cache_dir=CACHE_DIR)
        test_set = PVCBeatsDataset(test_ids, self.feature_types, cache_dir=CACHE_DIR)

        for ds, name in [(train_set, "train"), (test_set, "test")]:
            for i in tqdm(range(len(ds)), desc=f"Menghitung fitur ({name})", unit="beat"):
                ds[i]

        labels = np.array(train_set.labels())
        groups = np.array(train_set.patient_groups())
        skf = StratifiedGroupKFold(n_splits=self.num_folds, shuffle=True, random_state=config.RANDOM_SEED)

        pbar = tqdm(total=(self.num_folds + 1) * epochs, desc="Training", unit="epoch")

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels, groups)):
            tqdm.write(f"Fold {fold_idx + 1}/{self.num_folds} "
                       f"({len(set(groups[train_idx]))} pasien train, {len(set(groups[val_idx]))} pasien val)")
            model = self._fit(torch.utils.data.Subset(train_set, train_idx), labels[train_idx],
                              epochs, batch_size, learning_rate, pbar, f"{fold_idx + 1}/{self.num_folds}")
            fold_metrics = self._evaluate(model, self._loader(torch.utils.data.Subset(train_set, val_idx), batch_size, False))
            fold_results.append(fold_metrics)
            tqdm.write(f"  Fold result: macro_f1={fold_metrics['macro_f1']:.4f}, "
                       f"recall_lvot={fold_metrics['recall_lvot']:.4f}, auc={fold_metrics['auc']:.4f}\n")

        tqdm.write(f"Final: train di {len(set(groups))} pasien, uji di {len(test_ids)} pasien test")
        model = self._fit(train_set, labels, epochs, batch_size, learning_rate, pbar, "final")
        test_metrics = self._evaluate(model, self._loader(test_set, batch_size, False))
        pbar.close()

        return fold_results, test_metrics

    def _loader(self, dataset, batch_size, shuffle):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _fit(self, dataset, labels, epochs, batch_size, learning_rate, pbar, tag):
        model = self._build_model()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        counts = np.bincount(labels, minlength=2)
        class_weights = torch.tensor(len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loader = self._loader(dataset, batch_size, True)

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(model, loader, optimizer, criterion)
            pbar.set_postfix(fold=tag, loss=f"{train_loss:.4f}")
            pbar.update(1)
            if epoch % max(1, epochs // 3) == 0:
                tqdm.write(f"  Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}")
        return model

    def _build_model(self):
        model = BaselineCNN() if not self.feature_types else FusionCNN(self.feature_types)
        return model.to(self.device)

    def _forward(self, model, batch_device):
        if isinstance(model, BaselineCNN):
            return model(batch_device["raw"])
        return model(batch_device)

    def _to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _train_one_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            batch_device = self._to_device(batch)
            loss = criterion(self._forward(model, batch_device), batch_device["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate(self, model, loader):
        model.eval()
        predictions, targets, scores = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch_device = self._to_device(batch)
                probs = torch.softmax(self._forward(model, batch_device), dim=1)
                predictions.extend(probs.argmax(dim=1).cpu().numpy())
                scores.extend(probs[:, 1].cpu().numpy())
                targets.extend(batch_device["label"].cpu().numpy())
        return compute_metrics(targets, predictions, scores)
