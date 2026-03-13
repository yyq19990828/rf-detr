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

import contextlib
import copy
import time
from io import StringIO
from pathlib import Path
from typing import Any, List, Optional, Union

import torch.utils.data
import torchvision
from pycocotools.coco import COCO

from rfdetr.datasets.coco import build_coco, build_roboflow_from_coco
from rfdetr.datasets.o365 import build_o365
from rfdetr.datasets.yolo import YoloDetection, build_roboflow_from_yolo
from rfdetr.util.logger import get_logger

logger = get_logger()


def _build_coco_api_with_offset(coco_api: Any, image_id_offset: int) -> COCO:
    """Convert a dataset's COCO-like API to a real COCO object with offset image IDs."""
    dataset_dict = copy.deepcopy(coco_api.dataset)

    if image_id_offset:
        for image in dataset_dict.get("images", []):
            image["id"] += image_id_offset
        for annotation in dataset_dict.get("annotations", []):
            annotation["image_id"] += image_id_offset

    merged_coco = COCO()
    merged_coco.dataset = dataset_dict
    with contextlib.redirect_stdout(StringIO()):
        merged_coco.createIndex()

    label2cat = getattr(coco_api, "label2cat", None)
    if label2cat is not None:
        merged_coco.label2cat = copy.deepcopy(label2cat)

    return merged_coco


def _merge_coco_apis(coco_apis: List[COCO]) -> Optional[COCO]:
    """Merge multiple COCO objects into one for ConcatDataset evaluation."""
    if not coco_apis:
        return None

    merged_dataset = {
        "info": copy.deepcopy(coco_apis[0].dataset.get("info", {})),
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(coco_apis[0].dataset.get("categories", [])),
    }
    for coco_api in coco_apis:
        merged_dataset["images"].extend(copy.deepcopy(coco_api.dataset.get("images", [])))
        merged_dataset["annotations"].extend(copy.deepcopy(coco_api.dataset.get("annotations", [])))

    merged_coco = COCO()
    merged_coco.dataset = merged_dataset
    with contextlib.redirect_stdout(StringIO()):
        merged_coco.createIndex()

    label2cat = getattr(coco_apis[0], "label2cat", None)
    if label2cat is not None:
        merged_coco.label2cat = copy.deepcopy(label2cat)

    return merged_coco


class _ImageIdOffsetDataset(torch.utils.data.Dataset):
    """Dataset wrapper that makes image IDs globally unique across concatenated datasets."""

    def __init__(self, dataset: torch.utils.data.Dataset, image_id_offset: int) -> None:
        self.dataset = dataset
        self.image_id_offset = image_id_offset
        coco_api = get_coco_api_from_dataset(dataset)
        self.coco = _build_coco_api_with_offset(coco_api, image_id_offset) if coco_api is not None else None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        image, target = self.dataset[index]
        target = copy.deepcopy(target)
        image_id = target["image_id"]

        if hasattr(image_id, "clone"):
            target["image_id"] = image_id.clone()
            target["image_id"] += self.image_id_offset
        else:
            target["image_id"] = image_id + self.image_id_offset

        return image, target

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def _normalize_dataset_dirs(dataset_dir: Union[str, List[str]]) -> List[str]:
    """Normalise ``dataset_dir`` to a list of directory strings."""
    if isinstance(dataset_dir, (str, Path)):
        return [str(dataset_dir)]
    return [str(d) for d in dataset_dir]


def get_coco_api_from_dataset(dataset: torch.utils.data.Dataset) -> Optional[Any]:
    """Return the COCO API object from a dataset, handling ConcatDataset."""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        if hasattr(dataset, "coco"):
            return dataset.coco
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
    coco_apis = []
    next_image_id_offset = 0
    for d in dirs:
        logger.info("Loading split '%s' from %s ...", image_set, d)
        start = time.perf_counter()
        single_args = copy.copy(args)
        single_args.dataset_dir = d
        ds = builder_fn(image_set, single_args, resolution)
        wrapped_ds = _ImageIdOffsetDataset(ds, next_image_id_offset)
        datasets.append(wrapped_ds)
        if wrapped_ds.coco is not None:
            coco_apis.append(wrapped_ds.coco)
            image_ids = wrapped_ds.coco.getImgIds()
            next_image_id_offset = (max(image_ids) + 1) if image_ids else next_image_id_offset
        else:
            next_image_id_offset += len(wrapped_ds)
        elapsed = time.perf_counter() - start
        logger.info("Loaded %d samples from %s/%s in %.2fs", len(ds), d, image_set, elapsed)

    merged = torch.utils.data.ConcatDataset(datasets)
    merged.coco = _merge_coco_apis(coco_apis)
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
