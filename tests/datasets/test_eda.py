# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for COCO annotation EDA generation."""

import json
from pathlib import Path
from typing import Any

import torch

from rfdetr.datasets.eda import run_eda


class _MockCoco:
    """Minimal COCO-like object exposing fields used by ``run_eda``."""

    def __init__(
        self, imgs: dict[int, dict[str, Any]], cats: dict[int, dict[str, Any]], anns: dict[int, dict[str, Any]]
    ) -> None:
        self.imgs = imgs
        self.cats = cats
        self.anns = anns
        self.dataset = {
            "images": list(imgs.values()),
            "categories": list(cats.values()),
            "annotations": list(anns.values()),
        }
        self.imgToAnns = {}
        for ann in anns.values():
            image_id = ann["image_id"]
            if image_id not in self.imgToAnns:
                self.imgToAnns[image_id] = []
            self.imgToAnns[image_id].append(ann)


class _MockDataset(torch.utils.data.Dataset):
    """Minimal dataset that only exposes ``.coco`` for EDA."""

    def __init__(self, coco: _MockCoco) -> None:
        self.coco = coco

    def __len__(self) -> int:
        return len(self.coco.imgs)

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


def _load_summary(path: Path) -> dict[str, Any]:
    """Load a JSON summary file from disk."""
    return json.loads(path.read_text())


def test_run_eda_creates_expected_files_and_summary(tmp_path: Path) -> None:
    """run_eda writes all plot files and a complete summary JSON."""
    coco = _MockCoco(
        imgs={
            1: {"id": 1, "width": 100, "height": 100},
            2: {"id": 2, "width": 200, "height": 100},
        },
        cats={
            1: {"id": 1, "name": "cat"},
            2: {"id": 2, "name": "dog"},
        },
        anns={
            1: {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400},
            2: {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 20, 30, 40], "area": 1200},
            3: {"id": 3, "image_id": 2, "category_id": 1, "bbox": [0, 0, 40, 20], "area": 800},
        },
    )
    dataset = _MockDataset(coco)

    output_dir = tmp_path / "eda"
    result = run_eda(dataset=dataset, output_dir=output_dir, split="train")

    assert result == output_dir
    assert (output_dir / "train_class_distribution.png").exists()
    assert (output_dir / "train_bbox_area_distribution.png").exists()
    assert (output_dir / "train_bbox_aspect_ratio.png").exists()
    summary_path = output_dir / "train_eda_summary.json"
    assert summary_path.exists()

    summary = _load_summary(summary_path)
    assert summary["total_images"] == 2
    assert summary["total_annotations"] == 3
    assert summary["num_classes"] == 2
    assert summary["class_counts"] == {"cat": 2, "dog": 1}

    assert set(summary["bbox_area_stats"].keys()) == {"min", "max", "mean", "median"}
    assert set(summary["bbox_aspect_ratio_stats"].keys()) == {"min", "max", "mean", "median"}
    assert summary["bbox_area_stats"]["min"] > 0
    assert summary["bbox_area_stats"]["max"] >= summary["bbox_area_stats"]["min"]


def test_run_eda_handles_empty_annotations(tmp_path: Path) -> None:
    """run_eda handles datasets with no annotations without errors."""
    coco = _MockCoco(
        imgs={1: {"id": 1, "width": 128, "height": 128}},
        cats={1: {"id": 1, "name": "object"}},
        anns={},
    )
    dataset = _MockDataset(coco)

    output_dir = tmp_path / "empty"
    run_eda(dataset=dataset, output_dir=output_dir, split="train")

    summary = _load_summary(output_dir / "train_eda_summary.json")
    assert summary["total_images"] == 1
    assert summary["total_annotations"] == 0
    assert summary["class_counts"] == {"object": 0}
    assert summary["bbox_area_stats"] == {"min": None, "max": None, "mean": None, "median": None}
    assert summary["bbox_aspect_ratio_stats"] == {"min": None, "max": None, "mean": None, "median": None}


def test_run_eda_single_class_counts_unique_images(tmp_path: Path) -> None:
    """class_counts tracks number of images per class, not raw annotation count."""
    coco = _MockCoco(
        imgs={1: {"id": 1, "width": 64, "height": 64}},
        cats={1: {"id": 1, "name": "single"}},
        anns={
            1: {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100},
            2: {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 20], "area": 200},
        },
    )
    dataset = _MockDataset(coco)

    output_dir = tmp_path / "single"
    run_eda(dataset=dataset, output_dir=output_dir, split="train")
    summary = _load_summary(output_dir / "train_eda_summary.json")

    assert summary["num_classes"] == 1
    assert summary["class_counts"] == {"single": 1}
