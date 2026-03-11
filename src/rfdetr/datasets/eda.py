# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Exploratory Data Analysis utilities for COCO-style datasets."""

import json
import math
import statistics
from pathlib import Path
from typing import Any, Union

import torch

from rfdetr.util.logger import get_logger

logger = get_logger()


def _compute_stats(values: list[float]) -> dict[str, float | None]:
    """Compute basic summary statistics for a numeric list.

    Args:
        values: Numeric values.

    Returns:
        Dictionary with ``min``, ``max``, ``mean``, and ``median``.
        Returns ``None`` for all keys when ``values`` is empty.
    """
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


def run_eda(
    dataset: torch.utils.data.Dataset,
    output_dir: Union[str, Path],
    split: str = "train",
) -> Path:
    """Generate EDA artifacts from a COCO-style dataset.

    This function only reads the COCO annotation metadata exposed via
    ``dataset.coco`` and does not iterate over image pixels.

    Args:
        dataset: Dataset exposing a ``.coco`` object with ``imgs``, ``cats``,
            and ``anns`` dictionaries.
        output_dir: Directory where EDA plots and JSON summary are written.
        split: Split name used in output file names.

    Returns:
        Path to the output directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coco = getattr(dataset, "coco", None)
    if coco is None:
        raise ValueError("Dataset must expose a COCO API object via dataset.coco.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Running EDA for split '%s'", split)

    total_images = len(coco.imgs)
    total_annotations = len(coco.anns)
    num_classes = len(coco.cats)

    class_annotation_counts: dict[int, int] = {cat_id: 0 for cat_id in coco.cats.keys()}
    bbox_area_fractions: list[float] = []
    bbox_aspect_ratios: list[float] = []

    for ann in coco.anns.values():
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        image_info = coco.imgs.get(image_id)
        if image_info is None:
            continue

        image_width = image_info.get("width", 0)
        image_height = image_info.get("height", 0)
        image_area = float(image_width) * float(image_height)
        if image_area <= 0:
            continue

        if category_id in class_annotation_counts:
            class_annotation_counts[category_id] += 1

        bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        try:
            bbox_width = float(bbox[2])
            bbox_height = float(bbox[3])
        except (TypeError, ValueError, IndexError):
            continue

        ann_area = ann.get("area")
        if ann_area is not None:
            try:
                bbox_area = float(ann_area)
            except (TypeError, ValueError):
                bbox_area = bbox_width * bbox_height
        else:
            bbox_area = bbox_width * bbox_height
        if bbox_area >= 0 and math.isfinite(bbox_area):
            bbox_area_fraction = bbox_area / image_area
            if math.isfinite(bbox_area_fraction):
                bbox_area_fractions.append(bbox_area_fraction)

        if bbox_width > 0 and bbox_height > 0:
            bbox_aspect_ratio = bbox_width / bbox_height
            if math.isfinite(bbox_aspect_ratio):
                bbox_aspect_ratios.append(bbox_aspect_ratio)

    class_counts: dict[str, int] = {}
    for cat_id, category in sorted(coco.cats.items(), key=lambda item: item[1].get("name", "")):
        class_name = category.get("name", str(cat_id))
        class_counts[class_name] = class_annotation_counts.get(cat_id, 0)

    sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

    class_plot_path = output_path / f"{split}_class_distribution.png"
    plt.figure(figsize=(10, 6))
    if sorted_classes:
        labels = [item[0] for item in sorted_classes]
        values = [item[1] for item in sorted_classes]
        plt.barh(labels, values)
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, "No classes found", ha="center", va="center")
    plt.xlabel("Number of Boxes")
    plt.ylabel("Class")
    plt.title(f"{split.capitalize()} Class Distribution")
    plt.tight_layout()
    plt.savefig(class_plot_path, dpi=200)
    plt.close()

    area_plot_path = output_path / f"{split}_bbox_area_distribution.png"
    plt.figure(figsize=(10, 6))
    if bbox_area_fractions:
        plt.hist(bbox_area_fractions, bins=40)
    else:
        plt.text(0.5, 0.5, "No bounding boxes found", ha="center", va="center")
    plt.xlabel("BBox Area Fraction (BBox Area / Image Area)")
    plt.ylabel("Count")
    plt.title(f"{split.capitalize()} Bounding Box Area Distribution")
    plt.tight_layout()
    plt.savefig(area_plot_path, dpi=200)
    plt.close()

    aspect_plot_path = output_path / f"{split}_bbox_aspect_ratio.png"
    plt.figure(figsize=(10, 6))
    if bbox_aspect_ratios:
        plt.hist(bbox_aspect_ratios, bins=40)
        min_ratio = min(bbox_aspect_ratios)
        max_ratio = max(bbox_aspect_ratios)
        if min_ratio > 0 and max_ratio > min_ratio:
            plt.xlim(min_ratio, max_ratio)
    else:
        plt.text(0.5, 0.5, "No bounding boxes found", ha="center", va="center")
        plt.xlim(0.1, 10.0)
    plt.xscale("log")
    plt.xlabel("BBox Aspect Ratio (width / height)")
    plt.ylabel("Count")
    plt.title(f"{split.capitalize()} Bounding Box Aspect Ratio")
    plt.tight_layout()
    plt.savefig(aspect_plot_path, dpi=200)
    plt.close()

    summary = {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "num_classes": num_classes,
        "class_counts": class_counts,
        "bbox_area_stats": _compute_stats(bbox_area_fractions),
        "bbox_aspect_ratio_stats": _compute_stats(bbox_aspect_ratios),
    }

    summary_path = output_path / f"{split}_eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info("EDA artifacts for split '%s' saved to: %s", split, output_path.resolve())

    return output_path
