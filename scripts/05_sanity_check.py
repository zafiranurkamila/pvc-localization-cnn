"""Full pipeline sanity check: load → segment → extract → dataset → model forward pass.

Usage:
    python scripts/05_sanity_check.py --n-patients 3 --scenario psd_wavelet_hos
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.dataset import PVCBeatsDataset, create_train_val_test_split
from pvc_localization.models.fusion import FusionCNN, BaselineCNN
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=3)
    parser.add_argument("--scenario", type=str, default="psd_wavelet_hos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids = create_train_val_test_split()
    sample_ids = train_ids[:args.n_patients]

    feature_types = config.FEATURE_SCENARIOS[args.scenario]
    print(f"\nScenario: {args.scenario}")
    print(f"Features: {feature_types if feature_types else 'none (baseline)'}")
    print(f"Sample patients: {sample_ids}")

    print("\nBuilding dataset...")
    dataset = PVCBeatsDataset(sample_ids, feature_types)
    print(f"Total beats loaded: {len(dataset)}")

    print("\nBuilding model...")
    if feature_types:
        model = FusionCNN(feature_types).to(device)
        print(f"FusionCNN: {sum(p.numel() for p in model.parameters())} parameters")
    else:
        model = BaselineCNN().to(device)
        print(f"BaselineCNN: {sum(p.numel() for p in model.parameters())} parameters")

    print("\nRunning forward pass...")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if feature_types:
                logits = model(batch_device)
            else:
                logits = model(batch_device["psd"])

            print(f"  Batch {batch_idx + 1}: logits shape {logits.shape}, predictions {logits.argmax(dim=1).cpu().numpy()}")

            if batch_idx >= 1:
                break

    print("\n✅ Sanity check passed! Pipeline is ready for training on GPU.")


if __name__ == "__main__":
    main()
