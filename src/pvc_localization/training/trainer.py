"""K-Fold CV training loop for CNN models."""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from pvc_localization import config
from pvc_localization.data.dataset import PVCBeatsDataset, create_train_val_test_split
from pvc_localization.models.fusion import FusionCNN, BaselineCNN
from pvc_localization.evaluation.metrics import compute_metrics

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


class CVTrainer:
    """Manage K-fold cross-validation training."""

    def __init__(self, feature_types: list[str], num_folds: int = 5, device: str = None):
        self.feature_types = feature_types
        self.num_folds = num_folds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        train_ids: list[int],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        """Run K-fold CV on training data.

        Returns:
            fold_results: list of dicts with metrics per fold
        """
        fold_results = []

        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=config.RANDOM_SEED)
        dataset = PVCBeatsDataset(train_ids, self.feature_types, cache_dir=CACHE_DIR)

        labels = [dataset[i]["label"] for i in tqdm(range(len(dataset)), desc="Menghitung fitur", unit="beat")]

        pbar = tqdm(total=self.num_folds * epochs, desc="Training", unit="epoch")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            tqdm.write(f"Fold {fold_idx + 1}/{self.num_folds}")

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            model = self._build_model()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                train_loss = self._train_one_epoch(model, train_loader, optimizer, criterion)
                val_metrics = self._evaluate(model, val_loader)
                pbar.set_postfix(fold=f"{fold_idx + 1}/{self.num_folds}",
                                 loss=f"{train_loss:.4f}", val_acc=f"{val_metrics['accuracy']:.4f}")
                pbar.update(1)
                if epoch % max(1, epochs // 3) == 0:
                    tqdm.write(f"  Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_acc={val_metrics['accuracy']:.4f}")

            fold_metrics = self._evaluate(model, val_loader)
            fold_results.append(fold_metrics)
            tqdm.write(f"  Fold result: {fold_metrics}\n")

        pbar.close()
        return fold_results

    def _build_model(self):
        if self.feature_types == []:
            model = BaselineCNN()
        else:
            model = FusionCNN(self.feature_types)
        return model.to(self.device)

    def _forward(self, model, batch_device):
        if isinstance(model, BaselineCNN):
            return model(batch_device["psd"])
        return model(batch_device)

    def _train_one_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = self._forward(model, batch_device)

            labels = batch_device["label"].to(self.device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(self, model, loader):
        model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = self._forward(model, batch_device)

                preds = logits.argmax(dim=1).cpu().numpy()
                labels = batch_device["label"].cpu().numpy()

                predictions.extend(preds)
                targets.extend(labels)

        return compute_metrics(targets, predictions)