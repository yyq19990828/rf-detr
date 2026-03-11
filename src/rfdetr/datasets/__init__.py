# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import torch.utils.data
import torchvision

from rfdetr.datasets.coco import build_coco, build_roboflow_from_coco
from rfdetr.datasets.o365 import build_o365
from rfdetr.datasets.yolo import YoloDetection, build_roboflow_from_yolo
from rfdetr.util.logger import get_logger

logger = get_logger()


def _normalize_dataset_dirs(dataset_dir: Union[str, List[str]]) -> List[str]:
    """Normalise ``dataset_dir`` to a list of directory strings."""
    if isinstance(dataset_dir, (str, Path)):
        return [str(dataset_dir)]
    return [str(d) for d in dataset_dir]


def get_coco_api_from_dataset(dataset: torch.utils.data.Dataset) -> Optional[Any]:
    """Return the COCO API object from a dataset, handling ConcatDataset."""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        if dataset.datasets:
            return get_coco_api_from_dataset(dataset.datasets[0])
        return None
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    if isinstance(dataset, YoloDetection):
        return dataset.coco
    return None


def detect_roboflow_format(dataset_dir: Path) -> str:
    """Detect if a Roboflow dataset is in COCO or YOLO format.

    Args:
        dataset_dir: Path to the Roboflow dataset root directory

    Returns:
        'coco' if COCO format detected, 'yolo' if YOLO format detected

    Raises:
        ValueError: If neither format is detected
    """
    # Check for COCO format: look for _annotations.coco.json in train folder
    coco_annotation = dataset_dir / "train" / "_annotations.coco.json"
    if coco_annotation.exists():
        return "coco"

    # Check for YOLO format: look for data.yaml or data.yml and train/images folder
    yolo_data_file_yaml = dataset_dir / "data.yaml"
    yolo_data_file_yml = dataset_dir / "data.yml"
    yolo_images_dir = dataset_dir / "train" / "images"
    if (yolo_data_file_yaml.exists() or yolo_data_file_yml.exists()) and yolo_images_dir.exists():
        return "yolo"

    raise ValueError(
        f"Could not detect dataset format in {dataset_dir}. "
        f"Expected either COCO format (train/_annotations.coco.json) "
        f"or YOLO format (data.yaml or data.yml + train/images/)"
    )


def _build_single_roboflow(image_set: str, args: Any, resolution: int) -> torch.utils.data.Dataset:
    """Build a single Roboflow dataset from ``args.dataset_dir`` (must be a single path)."""
    root = Path(args.dataset_dir)
    assert root.exists(), f"provided Roboflow path {root} does not exist"

    dataset_format = detect_roboflow_format(root)

    if dataset_format == "coco":
        return build_roboflow_from_coco(image_set, args, resolution)
    return build_roboflow_from_yolo(image_set, args, resolution)


def _build_multi_dir(
    image_set: str,
    args: Any,
    resolution: int,
    builder_fn: Any,
) -> torch.utils.data.Dataset:
    """Build from one or more directories, merging via ConcatDataset when needed."""
    dirs = _normalize_dataset_dirs(args.dataset_dir)

    if len(dirs) == 1:
        single_args = copy.copy(args)
        single_args.dataset_dir = dirs[0]
        return builder_fn(image_set, single_args, resolution)

    datasets = []
    for d in dirs:
        logger.info("Loading split '%s' from %s ...", image_set, d)
        start = time.perf_counter()
        single_args = copy.copy(args)
        single_args.dataset_dir = d
        ds = builder_fn(image_set, single_args, resolution)
        datasets.append(ds)
        elapsed = time.perf_counter() - start
        logger.info("Loaded %d samples from %s/%s in %.2fs", len(ds), d, image_set, elapsed)

    merged = torch.utils.data.ConcatDataset(datasets)
    logger.info("Merged %d datasets (%d total samples) for split '%s'", len(datasets), len(merged), image_set)
    return merged


def build_roboflow(image_set: str, args: Any, resolution: int) -> torch.utils.data.Dataset:
    """Build a Roboflow dataset, auto-detecting COCO or YOLO format.

    Supports multiple dataset directories via ``args.dataset_dir`` as a list.
    """
    return _build_multi_dir(image_set, args, resolution, _build_single_roboflow)


def build_dataset(image_set: str, args: Any, resolution: int) -> torch.utils.data.Dataset:
    """Build a dataset for the given split.

    Supports multiple dataset directories for ``roboflow`` and ``yolo`` formats
    when ``args.dataset_dir`` is a list of paths.
    """
    if args.dataset_file == "coco":
        return build_coco(image_set, args, resolution)
    if args.dataset_file == "o365":
        return build_o365(image_set, args, resolution)
    if args.dataset_file == "roboflow":
        return build_roboflow(image_set, args, resolution)
    if args.dataset_file == "yolo":
        return _build_multi_dir(image_set, args, resolution, build_roboflow_from_yolo)
    raise ValueError(f"dataset {args.dataset_file} not supported")
