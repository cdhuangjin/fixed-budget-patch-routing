"""Pixel-level localization audit for the pretrained full patch route."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from evaluate_mvtec_patchcore import ResNetFeatures, collect_features, split_normal_records
from evaluate_mvtec_pro import pro_auc
from mvtec_ad import MVTecADIndex


def auroc(y, s):
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    device = torch.device(args.device)
    model = ResNetFeatures(device)
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    full_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    index = MVTecADIndex(args.data_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        train = index.samples(category, "train")
        test = index.samples(category, "test")
        fit, _ = split_normal_records(train, args.seed)
        fit_features, _, _ = collect_features(model, fit, 224, full_transform, args.batch_size, device, full=True)
        test_features, _, _ = collect_features(model, test, 224, full_transform, args.batch_size, device, full=True)
        bank = torch.cat([x for x in fit_features], dim=0).to(device)
        pixel_y, pixel_s = [], []
        pro_maps, pro_masks = [], []
        for record, sample in zip(test, test_features):
            side = int(round(sample.shape[0] ** 0.5))
            distances = torch.cdist(sample.to(device), bank).min(dim=1).values.reshape(1, 1, side, side)
            score_map = F.interpolate(
                distances, size=(224, 224), mode="bilinear", align_corners=False
            )[0, 0].detach().cpu().numpy()
            if record["mask_path"]:
                mask = Image.open(record["mask_path"]).convert("L").resize((224, 224), Image.Resampling.NEAREST)
                mask_array = np.asarray(mask) > 0
                pixel_y.extend(mask_array.reshape(-1).astype(np.uint8).tolist())
                pixel_s.extend(score_map.reshape(-1).tolist())
                pro_maps.append(score_map)
                pro_masks.append(mask_array)
            elif not record["is_anomaly"]:
                pixel_y.extend([0] * score_map.size)
                pixel_s.extend(score_map.reshape(-1).tolist())
                pro_maps.append(score_map)
                pro_masks.append(np.zeros_like(score_map, dtype=bool))
        result = {
            "category": category,
            "pixel_auroc": auroc(pixel_y, pixel_s),
            "pro_auc_max_fpr_0.30": pro_auc(pro_maps, pro_masks, max_fpr=0.30),
            "pixels": len(pixel_y),
            "pro_images": len(pro_maps),
        }
        (out / f"{category}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
