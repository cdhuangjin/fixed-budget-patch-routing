"""Train category-specific binary Sparse/Full models with synthetic defects."""

import argparse
import random
from pathlib import Path

import torch

# === RTX 5060 Blackwell 优化 ===
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from gpu_optimize import setup_rtx5060_optimizations
    setup_rtx5060_optimizations()
except ImportError:
    pass
# === 优化结束 ===
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mvtec_ad import MVTecADIndex
from smoke_train import build_model


class SyntheticDefectDataset(Dataset):
    def __init__(self, records, image_size=32, seed=17):
        self.records = list(records)
        self.image_size = image_size
        self.seed = seed
        # Decode/resize each source image once.  The previous implementation
        # reopened the same files for every epoch, making this experiment I/O
        # bound despite having a CUDA device available.
        normal_images = []
        defect_images = []
        for index, record in enumerate(self.records):
            image = Image.open(record["image_path"]).convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).clone()
            tensor = values.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
            normal_images.append(tensor)
            generator = torch.Generator().manual_seed(self.seed + len(self.records) + index)
            size = max(2, self.image_size // 5)
            y = int(torch.randint(0, self.image_size - size + 1, (1,), generator=generator))
            x = int(torch.randint(0, self.image_size - size + 1, (1,), generator=generator))
            noise = torch.rand((3, size, size), generator=generator)
            defect = tensor.clone()
            defect[:, y:y + size, x:x + size] = noise
            defect_images.append(defect)
        self.images = torch.stack(normal_images + defect_images)
        self.labels = torch.cat(
            [torch.zeros(len(normal_images), dtype=torch.long),
             torch.ones(len(defect_images), dtype=torch.long)]
        )

    def __len__(self):
        return len(self.records) * 2

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def train_one(method, loader, device, epochs, seed):
    torch.manual_seed(seed)
    model = build_model(method, image_size=32, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.cpu().state_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    random.seed(args.seed)
    records = MVTecADIndex(args.data_root).samples(args.category, "train")
    records = [record for record in records if not record["is_anomaly"]]
    loader = DataLoader(SyntheticDefectDataset(records, seed=args.seed), batch_size=args.batch_size, shuffle=True, num_workers=0)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for method, name in (("fixed_sparse", "sparse"), ("full", "full")):
        state = train_one(method, loader, torch.device(args.device), args.epochs, args.seed)
        torch.save(state, output_root / f"{args.category}_{name}.pt")
    print(f"trained {args.category} with {len(records)} normal images")


if __name__ == "__main__":
    main()
