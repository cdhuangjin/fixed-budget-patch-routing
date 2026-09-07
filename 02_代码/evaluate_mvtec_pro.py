"""Region-level PRO evaluation for MVTec score maps.

This module is deliberately independent of model inference. It consumes score
maps and masks so that threshold calibration cannot accidentally use test masks.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def connected_components(binary):
    binary = np.asarray(binary, dtype=bool)
    visited = np.zeros(binary.shape, dtype=bool)
    components = []
    height, width = binary.shape
    for row in range(height):
        for col in range(width):
            if not binary[row, col] or visited[row, col]:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            pixels = []
            while queue:
                current_row, current_col = queue.popleft()
                pixels.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if 0 <= next_row < height and 0 <= next_col < width:
                        if binary[next_row, next_col] and not visited[next_row, next_col]:
                            visited[next_row, next_col] = True
                            queue.append((next_row, next_col))
            components.append(np.asarray(pixels, dtype=np.int64))
    return components


def pro_curve(score_maps, masks, thresholds=None):
    score_maps = [np.asarray(value, dtype=float) for value in score_maps]
    masks = [np.asarray(value, dtype=bool) for value in masks]
    if len(score_maps) != len(masks) or not score_maps:
        raise ValueError("score_maps and masks must be non-empty and paired")
    if any(scores.shape != mask.shape for scores, mask in zip(score_maps, masks)):
        raise ValueError("score map and mask shapes must match")
    if thresholds is None:
        finite_scores = np.concatenate([scores[np.isfinite(scores)] for scores in score_maps])
        thresholds = np.unique(np.quantile(finite_scores, np.linspace(0.0, 1.0, 101)))
    normal_pixels = sum(int((~mask).sum()) for mask in masks)
    if normal_pixels == 0:
        raise ValueError("PRO requires at least one normal pixel")
    regions = []
    for mask in masks:
        regions.extend((mask, component) for component in connected_components(mask))
    points = []
    for threshold in np.asarray(thresholds, dtype=float):
        false_positive = 0
        overlap_values = []
        for scores, mask in zip(score_maps, masks):
            predicted = scores >= threshold
            false_positive += int((predicted & ~mask).sum())
            for region_mask, component in regions:
                if region_mask is mask:
                    rows, cols = component[:, 0], component[:, 1]
                    overlap_values.append(float(predicted[rows, cols].mean()))
        points.append((false_positive / normal_pixels, float(np.mean(overlap_values)) if overlap_values else 0.0))
    points.sort(key=lambda point: point[0])
    return np.asarray(points, dtype=float)


def pro_auc(score_maps, masks, max_fpr=0.3, thresholds=None):
    if not 0.0 < float(max_fpr) <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    curve = pro_curve(score_maps, masks, thresholds=thresholds)
    clipped = curve[curve[:, 0] <= float(max_fpr)]
    if clipped.size == 0:
        return 0.0
    if clipped[0, 0] > 0.0:
        clipped = np.vstack(([0.0, clipped[0, 1]], clipped))
    if clipped[-1, 0] < float(max_fpr):
        clipped = np.vstack((clipped, [float(max_fpr), clipped[-1, 1]]))
    return float(np.trapezoid(clipped[:, 1], clipped[:, 0]) / float(max_fpr))
