import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision import transforms
from mpdd import MPDDIndex
from selective_routes import strict_quota_combined_scores

class ImageRows(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        return self.transform(Image.open(r["image_path"]).convert("RGB")), i

class Featurizer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = torch.nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.l1 = base.layer1
        self.l2 = base.layer2
        self.l3 = base.layer3
        self.device = torch.device(device)
        self.to(self.device).eval()
    @torch.inference_mode()
    def fast(self, x):
        x = x.to(self.device, non_blocking=True)
        x = self.l2(self.l1(self.stem(x)))
        return torch.nn.functional.normalize(x.mean((2,3)), dim=1)
    @torch.inference_mode()
    def full(self, x):
        x = x.to(self.device, non_blocking=True)
        x = self.l3(self.l2(self.l1(self.stem(x))))
        return torch.nn.functional.normalize(x.flatten(2).transpose(1,2), dim=2)

def split_normal(records, seed=17):
    normal = [r for r in records if not r["is_anomaly"]]
    rng = random.Random(seed)
    rng.shuffle(normal)
    cut = max(1, int(round(len(normal)*0.8)))
    return normal[:cut], normal[cut:]

def fast_scores(features, bank):
    return torch.cdist(features.cpu(), bank).amin(1).numpy()

def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--fallback-budget", type=float, default=0.25)
    args = p.parse_args()
    device = torch.device(args.device)
    fast_transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    index = MPDDIndex(args.data_root)
    model = Featurizer(device)
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    results = []
    for cat in args.categories:
        train = index.samples(cat, "train")
        test = index.samples(cat, "test")
        train_n, val_n = split_normal(train, args.seed)
        bank_loader = DataLoader(ImageRows(train_n, fast_transform), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type=="cuda")
        bank = []
        for images, _ in bank_loader:
            bank.append(model.fast(images).cpu())
        bank = torch.cat(bank)
        loader = DataLoader(ImageRows(test, fast_transform), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type=="cuda")
        scores, labels = [], []
        for images, idxs in loader:
            f = model.fast(images)
            sc = fast_scores(f, bank)
            for i, s in zip(idxs.tolist(), sc):
                scores.append(float(s))
                labels.append(int(test[i]["is_anomaly"]))
        scores = np.array(scores); labels = np.array(labels)
        boost = 1.5
        risk_scores, rand_scores, risk_route, random_route = strict_quota_combined_scores(
            scores, args.fallback_budget, args.seed, boost
        )
        rec = {
            "category": cat,
            "fast_only_auroc": float(auroc(labels, scores)),
            "risk_combined_auroc": float(auroc(labels, risk_scores)),
            "random_combined_auroc": float(auroc(labels, rand_scores)),
            "risk_delta": float(auroc(labels, risk_scores) - auroc(labels, rand_scores)),
            "budget": args.fallback_budget,
            "seed": args.seed,
            "fallback_rate": risk_route.actual_rate,
            "risk_count": risk_route.actual_fallback_count,
            "total": len(labels),
            "threshold": None,
            "route": risk_route.as_dict(),
            "random_route": random_route.as_dict(),
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fast_avg = float(np.mean([r["fast_only_auroc"] for r in results]))
    risk_avg = float(np.mean([r["risk_combined_auroc"] for r in results]))
    random_avg = float(np.mean([r["random_combined_auroc"] for r in results]))
    delta_avg = float(np.mean([r["risk_delta"] for r in results]))
    print(f"\nFast avg: {fast_avg:.4f}")
    print(f"Risk combined avg: {risk_avg:.4f}")
    print(f"Random combined avg: {random_avg:.4f}")
    print(f"Risk-Random delta avg: {delta_avg:+.4f}")
    print(f"Risk > Random in {sum(1 for r in results if r['risk_delta']>0)}/{len(results)} categories")

if __name__ == "__main__":
    main()

