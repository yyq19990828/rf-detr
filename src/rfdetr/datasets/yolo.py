# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import gc
import hashlib
import os
import pickle
import sys
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

from rfdetr.datasets.coco import (
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import get_rank

logger = get_logger()


REQUIRED_YOLO_YAML_FILES = ["data.yaml", "data.yml"]
REQUIRED_SPLIT_DIRS = ["train", "valid"]
REQUIRED_DATA_SUBDIRS = ["images", "labels"]


def is_valid_yolo_dataset(dataset_dir: str) -> bool:
    """
    Checks if the specified dataset directory is in yolo format.

    We accept a dataset to be in yolo format if the following conditions are met:
    - The dataset_dir contains a data.yaml or data.yml file
    - The dataset_dir contains "train" and "valid" subdirectories, each containing "images" and "labels" subdirectories
    - The "test" subdirectory is optional

    Returns a boolean indicating whether the dataset is in correct yolo format.
    """
    contains_required_yolo_yaml = any(
        os.path.exists(os.path.join(dataset_dir, yaml_file)) for yaml_file in REQUIRED_YOLO_YAML_FILES
    )
    contains_required_split_dirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir)) for split_dir in REQUIRED_SPLIT_DIRS
    )
    contains_required_data_subdirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir, data_subdir))
        for split_dir in REQUIRED_SPLIT_DIRS
        for data_subdir in REQUIRED_DATA_SUBDIRS
    )
    return contains_required_yolo_yaml and contains_required_split_dirs and contains_required_data_subdirs


# ---------------------------------------------------------------------------
# Image file extensions recognised by the YOLO loader (same as supervision)
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = frozenset({".bmp", ".dng", ".jpg", ".jpeg", ".mpo", ".png", ".tif", ".tiff", ".webp"})

# Cache format version – bump when the on-disk layout changes so that stale
# caches are automatically regenerated.
_CACHE_VERSION = 5

# Thread count for parallel image scanning (matches ultralytics convention).
_NUM_THREADS = min(8, max(1, os.cpu_count() - 1))


def _hash_directory(directory: str, extensions: frozenset[str]) -> str:
    """Return a fast hash summarising the file listing of *directory*.

    The hash is built from sorted file names and their ``os.stat`` results
    (size + mtime) so that any addition, removal or modification of image /
    label files invalidates the cache.

    Args:
        directory: Path to directory to hash.
        extensions: Set of lowercase file extensions (with leading dot) to
            include.  Pass ``None`` to include **all** files.

    Returns:
        A hex-digest string representing the directory state.
    """
    h = hashlib.sha256()
    try:
        entries = sorted(os.listdir(directory))
    except FileNotFoundError:
        return ""
    for name in entries:
        if extensions and os.path.splitext(name)[1].lower() not in extensions:
            continue
        full = os.path.join(directory, name)
        try:
            st = os.stat(full)
            h.update(f"{name},{st.st_size},{st.st_mtime_ns}".encode())
        except OSError:
            h.update(name.encode())
    return h.hexdigest()


def _list_image_paths(images_directory: str) -> list[str]:
    """List image file paths in *images_directory*, sorted for determinism.

    Args:
        images_directory: Path to directory containing image files.

    Returns:
        Sorted list of absolute image file paths.
    """
    paths: list[str] = []
    for name in sorted(os.listdir(images_directory)):
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
            paths.append(os.path.join(images_directory, name))
    return paths


def _parse_yolo_label_line(line: str) -> tuple[int, list[str]] | None:
    """Parse a single YOLO label line, skipping invalid entries.

    Returns ``(class_id, remaining_values)`` on success, or ``None`` if the
    line should be skipped (e.g. non-numeric class ID).

    Args:
        line: A single line from a YOLO ``.txt`` annotation file.

    Returns:
        Tuple of (class_id, value_strings) or ``None`` when invalid.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
    except ValueError:
        # Non-numeric class ID (e.g. "slagcar") – skip with warning
        return None
    return class_id, parts[1:]


def _parse_yolo_annotations(
    lines: list[str],
    resolution_wh: tuple[int, int],
    force_masks: bool = False,
) -> tuple[sv.Detections, int]:
    """Parse YOLO annotation lines into a supervision ``Detections`` object.

    Invalid lines (non-numeric class IDs, too few values, etc.) are silently
    skipped and counted.

    Args:
        lines: Raw lines from a YOLO ``.txt`` annotation file.
        resolution_wh: ``(width, height)`` of the corresponding image.
        force_masks: Whether to generate mask arrays from polygon
            annotations.

    Returns:
        A tuple of ``(detections, n_skipped)`` where *n_skipped* is the
        number of lines that were ignored.
    """
    if not lines:
        return sv.Detections.empty(), 0

    w, h = resolution_wh
    class_ids: list[int] = []
    relative_xyxy: list[np.ndarray] = []
    relative_polygons: list[np.ndarray] = []
    n_skipped = 0

    # Determine if any line has polygon-style annotations (> 5 values)
    with_masks = force_masks or any(len(ln.split()) > 5 for ln in lines)

    for line in lines:
        parsed = _parse_yolo_label_line(line)
        if parsed is None:
            n_skipped += 1
            continue
        cid, values = parsed

        if len(values) == 4:
            # Standard bounding box: x_center y_center width height
            xc, yc, bw, bh = (float(v) for v in values)
            box = np.array(
                [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2],
                dtype=np.float32,
            )
            relative_xyxy.append(box)
            if with_masks:
                relative_polygons.append(
                    np.array(
                        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                        dtype=np.float32,
                    )
                )
        elif len(values) >= 6 and len(values) % 2 == 0:
            # Polygon / segmentation annotation
            polygon = np.array([float(v) for v in values], dtype=np.float32).reshape(-1, 2)
            xs, ys = polygon[:, 0], polygon[:, 1]
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            relative_xyxy.append(box)
            if with_masks:
                relative_polygons.append(polygon)
        else:
            n_skipped += 1
            continue

        class_ids.append(cid)

    if not class_ids:
        return sv.Detections.empty(), n_skipped

    class_id_arr = np.array(class_ids, dtype=int)
    xyxy_arr = np.array(relative_xyxy, dtype=np.float32)
    xyxy_arr *= np.array([w, h, w, h], dtype=np.float32)

    if not with_masks:
        return sv.Detections(class_id=class_id_arr, xyxy=xyxy_arr), n_skipped

    from supervision.detection.utils.converters import polygon_to_mask

    abs_polygons = [(p * np.array([w, h], dtype=np.float32)).astype(int) for p in relative_polygons]
    masks = np.array(
        [polygon_to_mask(polygon=p, resolution_wh=(w, h)) for p in abs_polygons],
        dtype=bool,
    )
    return (
        sv.Detections(class_id=class_id_arr, xyxy=xyxy_arr, mask=masks),
        n_skipped,
    )


def load_yolo_annotations_cached(
    images_directory_path: str,
    annotations_directory_path: str,
    data_yaml_path: str,
    force_masks: bool = False,
) -> tuple[sv.DetectionDataset, dict[str, tuple[int, int]]]:
    """Load a YOLO dataset with ``.cache`` file support.

    On first invocation the function scans every image (for dimensions) and
    every label file, then writes a ``.cache`` pickle next to the labels
    directory.  Subsequent calls with the **same** file listing load
    instantly from cache.

    Invalid label lines (non-numeric class IDs, malformed values) are
    automatically skipped and a summary warning is emitted.

    Args:
        images_directory_path: Path to the directory containing images.
        annotations_directory_path: Path to the directory containing YOLO
            ``.txt`` annotation files.
        data_yaml_path: Path to ``data.yaml`` with class name definitions.
        force_masks: If ``True``, load / generate segmentation masks for
            every annotation.

    Returns:
        A tuple of:
        - A :class:`supervision.DetectionDataset` instance.
        - A dict mapping image path to ``(width, height)`` in pixels.
    """
    from tqdm.auto import tqdm

    from rfdetr.util.misc import get_rank, is_dist_avail_and_initialized

    rank = get_rank()
    is_distributed = is_dist_avail_and_initialized()

    # ---- Determine cache path (inside labels dir as .cache) ----
    labels_dir = Path(annotations_directory_path)
    cache_path = labels_dir / ".cache"

    # ---- Compute directory hashes for freshness check --------------------
    img_hash = _hash_directory(images_directory_path, _IMAGE_EXTENSIONS)
    lbl_hash = _hash_directory(annotations_directory_path, frozenset({".txt"}))
    # Include resolved dataset roots in the cache fingerprint so copied or
    # remounted datasets do not reuse a cache that still points at stale
    # absolute image paths from another machine.
    images_root = str(Path(images_directory_path).resolve())
    labels_root = str(Path(annotations_directory_path).resolve())
    current_hash = f"v{_CACHE_VERSION}:{images_root}:{labels_root}:{img_hash}:{lbl_hash}"

    # ---- Helper: deserialise a validated cache dict -----------------------
    def _load_from_cache_data(
        cache: dict,
    ) -> tuple[sv.DetectionDataset, dict[str, tuple[int, int]]]:
        classes_c = cache["classes"]
        image_paths_c = cache["image_paths"]
        annotations_c = {ip: sv.Detections(**det_kwargs) for ip, det_kwargs in cache["annotations"].items()}
        image_sizes_c: dict[str, tuple[int, int]] = cache.get("image_sizes", {})
        logger.info("Loaded %d images from cache %s", len(image_paths_c), cache_path)
        return (
            sv.DetectionDataset(
                classes=classes_c,
                images=image_paths_c,
                annotations=annotations_c,
            ),
            image_sizes_c,
        )

    # ---- Helper: unpickle with GC disabled for large caches --------------
    def _fast_unpickle(data: bytes) -> dict:
        gc.disable()
        try:
            return pickle.loads(data)  # noqa: S301
        finally:
            gc.enable()

    # ---- Try loading from cache (all ranks) ------------------------------
    if cache_path.exists():
        try:
            cache = _fast_unpickle(cache_path.read_bytes())
            if cache.get("hash") == current_hash:
                return _load_from_cache_data(cache)
            logger.info("Cache %s is stale, regenerating…", cache_path)
        except Exception:
            logger.warning("Failed to load cache %s, regenerating…", cache_path)

    # ---- DDP: only rank 0 performs the full scan -------------------------
    if is_distributed and rank != 0:
        logger.info("Rank %d waiting for rank 0 to build cache %s …", rank, cache_path)
        for _wait in range(3600):
            time.sleep(1)
            if cache_path.exists():
                try:
                    cache = _fast_unpickle(cache_path.read_bytes())
                    if cache.get("hash") == current_hash:
                        return _load_from_cache_data(cache)
                except Exception:
                    pass
        raise RuntimeError(f"Rank {rank}: timed out waiting for rank 0 to write cache {cache_path}")

    # ---- Full scan (rank 0 or single-process) ----------------------------
    from supervision.utils.file import read_yaml_file

    data = read_yaml_file(file_path=data_yaml_path)
    names = data["names"]
    if isinstance(names, dict):
        classes = [names[key] for key in sorted(names.keys())]
    else:
        classes = list(names)

    image_paths = _list_image_paths(images_directory_path)
    annotations: dict[str, sv.Detections] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    total_skipped = 0
    n_corrupt = 0

    def _scan_single(image_path: str) -> tuple[str, sv.Detections | None, int, tuple[int, int] | None]:
        """Verify one image and parse its label file.

        Returns:
            Tuple of ``(image_path, detections_or_None, n_skipped_lines,
            (width, height)_or_None)``.  ``None`` values signal that the
            image should be excluded (corrupt / unreadable).
        """
        # -- Read image header (no pixel decode) for dimensions + verify --
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                img.verify()
        except Exception:
            return (image_path, None, 0, None)

        stem = Path(image_path).stem
        annotation_path = os.path.join(annotations_directory_path, f"{stem}.txt")

        if not os.path.exists(annotation_path):
            return (image_path, sv.Detections.empty(), 0, (w, h))

        try:
            with open(annotation_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except OSError:
            return (image_path, sv.Detections.empty(), 0, (w, h))

        res_wh: tuple[int, int] = (w, h) if force_masks else (1, 1)
        det, n_skip = _parse_yolo_annotations(lines, resolution_wh=res_wh, force_masks=force_masks)
        return (image_path, det, n_skip, (w, h))

    start_t = time.perf_counter()
    with ThreadPool(_NUM_THREADS) as pool:
        results = pool.imap(
            _scan_single,
            image_paths,
        )
        for image_path, det, n_skip, wh in tqdm(
            results, total=len(image_paths), desc=f"Scanning {labels_dir.name}", unit="img"
        ):
            if det is None:
                n_corrupt += 1
                continue
            annotations[image_path] = det
            if wh is not None:
                image_sizes[image_path] = wh
            total_skipped += n_skip

    elapsed = time.perf_counter() - start_t

    valid_image_paths = list(annotations.keys())

    logger.info(
        "Scanned %d images (%d valid, %d corrupt) in %.1fs (%s)",
        len(image_paths),
        len(valid_image_paths),
        n_corrupt,
        elapsed,
        labels_dir.name,
    )
    if total_skipped > 0:
        logger.warning(
            "Skipped %d invalid label lines in %s (non-numeric class ID or malformed)",
            total_skipped,
            labels_dir.name,
        )

    # ---- Serialise annotations for pickle --------------------------------
    serialisable_annotations = {}
    for ip, det in annotations.items():
        kwargs: dict[str, Any] = {"xyxy": det.xyxy, "class_id": det.class_id}
        if det.mask is not None:
            kwargs["mask"] = det.mask
        serialisable_annotations[ip] = kwargs

    cache_data = {
        "hash": current_hash,
        "classes": classes,
        "image_paths": valid_image_paths,
        "annotations": serialisable_annotations,
        "image_sizes": image_sizes,
    }

    try:
        cache_path.write_bytes(pickle.dumps(cache_data))
        logger.info("Saved cache → %s", cache_path)
    except OSError:
        logger.warning("Could not write cache file %s", cache_path)

    return (
        sv.DetectionDataset(classes=classes, images=valid_image_paths, annotations=annotations),
        image_sizes,
    )


class ConvertYolo:
    """
    Converts supervision Detections to the target dict format expected by RF-DETR.

    Args:
        include_masks: whether to include segmentation masks
        normalized_coords: when ``True``, ``detections.xyxy`` values are in
            [0, 1] range and will be scaled to pixel coordinates using the
            actual image dimensions at call time.

    Examples:
        >>> import numpy as np
        >>> import supervision as sv
        >>> from PIL import Image
        >>> # Create a sample image and target
        >>> image = Image.new("RGB", (100, 100))
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[10, 20, 30, 40]]),
        ...     class_id=np.array([0])
        ... )
        >>> target = {"image_id": 0, "detections": detections}
        >>> # Create converter
        >>> converter = ConvertYolo(include_masks=False)
        >>> # Call converter
        >>> img, result = converter(image, target)
        >>> sorted(result.keys())
        ['area', 'boxes', 'image_id', 'iscrowd', 'labels', 'orig_size', 'size']
        >>> result["boxes"].shape
        torch.Size([1, 4])
        >>> result["labels"].tolist()
        [0]
        >>> result["image_id"].tolist()
        [0]
    """

    def __init__(self, include_masks: bool = False, normalized_coords: bool = False):
        self.include_masks = include_masks
        self.normalized_coords = normalized_coords

    def __call__(self, image: Image.Image, target: dict) -> tuple:
        """
        Convert image and YOLO detections to RF-DETR format.

        Args:
            image: PIL Image
            target: dict with 'image_id' and 'detections' (sv.Detections)

        Returns:
            tuple of (image, target_dict)
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        detections = target["detections"]

        if len(detections) > 0:
            boxes = torch.from_numpy(detections.xyxy).to(torch.float32)
            classes = torch.from_numpy(detections.class_id).to(torch.int64)

            finite_mask = torch.isfinite(boxes).all(dim=1)
            boxes = boxes[finite_mask]
            classes = classes[finite_mask]

            if self.normalized_coords:
                boxes[:, 0::2] *= w
                boxes[:, 1::2] *= h
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)

        # clamp and filter
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target_out = {}
        target_out["boxes"] = boxes
        target_out["labels"] = classes
        target_out["image_id"] = image_id

        # compute area after clamp
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target_out["area"] = area

        iscrowd = torch.zeros((classes.shape[0],), dtype=torch.int64)
        target_out["iscrowd"] = iscrowd

        if self.include_masks:
            if detections.mask is not None and np.size(detections.mask) > 0:
                masks = torch.from_numpy(detections.mask[keep.cpu().numpy()]).to(torch.uint8)
                target_out["masks"] = masks
            else:
                target_out["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target_out["masks"] = target_out["masks"].bool()

        target_out["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_out["size"] = torch.as_tensor([int(h), int(w)])

        return image, target_out


class _MockSvDataset:
    """Mock supervision dataset for testing CocoLikeAPI."""

    classes = ["cat", "dog"]

    def __init__(self):
        import numpy as np
        import supervision as sv

        self.image_paths = ["img_0.jpg", "img_1.jpg"]
        self.annotations = {
            "img_0.jpg": sv.Detections(xyxy=np.array([[0, 20, 30, 40]]), class_id=np.array([0])),
            "img_1.jpg": sv.Detections(xyxy=np.array([[10, 20, 30, 40]]), class_id=np.array([1])),
        }

    def __len__(self):
        return 2


class CocoLikeAPI:
    """
    A minimal COCO-compatible API wrapper for YOLO datasets.

    This provides the necessary interface for CocoEvaluator to work with
    YOLO format datasets.

    Examples:
        >>> mock = _MockSvDataset()
        >>> sizes = {"img_0.jpg": (100, 100), "img_1.jpg": (100, 100)}
        >>> coco = CocoLikeAPI(mock.classes, mock, sizes)
        >>> # dataset structure
        >>> len(coco.dataset["images"]), len(coco.dataset["categories"]), len(coco.dataset["annotations"])
        (2, 2, 2)
        >>> # getAnnIds
        >>> coco.getAnnIds()
        [0, 1]
        >>> coco.getAnnIds(imgIds=[0])
        [0]
        >>> coco.getAnnIds(catIds=[1])
        [1]
        >>> # getCatIds
        >>> sorted(coco.getCatIds())
        [0, 1]
        >>> coco.getCatIds(catNms=["cat"])
        [0]
        >>> # getImgIds
        >>> sorted(coco.getImgIds())
        [0, 1]
        >>> coco.getImgIds(catIds=[0])
        [0]
        >>> # loadAnns
        >>> ann = coco.loadAnns([0])[0]
        >>> ann["category_id"], ann["image_id"]
        (0, 0)
        >>> # loadCats
        >>> coco.loadCats([0])[0]["name"]
        'cat'
        >>> len(coco.loadCats())
        2
        >>> # loadImgs
        >>> coco.loadImgs([1])[0]["file_name"]
        'img_1.jpg'
    """

    def __init__(
        self,
        classes: list,
        dataset: sv.DetectionDataset,
        image_sizes: dict[str, tuple[int, int]],
    ):
        self.classes = classes
        self.sv_dataset = dataset
        self.image_sizes = image_sizes

        # Build COCO dataset dict and all lookup indices in a single pass.
        self._build_all()

    def _build_all(self) -> None:
        """Build COCO-format dataset dict and all lookup indices in one pass.

        Merges what were previously separate build + index phases into a single
        traversal over images/annotations.  Key optimisations vs the naive
        approach:

        * ``catToImgs`` uses intermediate ``set`` (O(1) membership) instead of
          ``list`` with ``in`` check (O(n) per annotation → O(n²) total).
        * ``imgToAnns`` / ``anns`` / ``imgs`` dicts are populated during the
          build loop rather than via extra passes over the final list.
        * ``sv.xyxy_to_xywh`` is called once per image (batch), not per box.
        * Per-image ``xywh.tolist()`` bulk-converts the entire numpy array
          instead of calling ``float()`` 4× per annotation.
        """
        categories: list[dict[str, Any]] = [
            {"id": idx, "name": name, "supercategory": "none"} for idx, name in enumerate(self.classes)
        ]

        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        imgs: dict[int, dict[str, Any]] = {}
        anns: dict[int, dict[str, Any]] = {}
        img_to_anns: dict[int, list[dict[str, Any]]] = {}
        cat_to_imgs_set: dict[int, set[int]] = {idx: set() for idx in range(len(self.classes))}

        ann_id = 0
        for img_id, image_path in enumerate(self.sv_dataset.image_paths):
            detections = self.sv_dataset.annotations[image_path]
            w, h = self.image_sizes.get(image_path, (0, 0))

            img_dict: dict[str, Any] = {
                "id": img_id,
                "file_name": str(image_path),
                "height": h,
                "width": w,
            }
            images.append(img_dict)
            imgs[img_id] = img_dict

            n_det = len(detections)
            if n_det == 0:
                img_to_anns[img_id] = []
                continue

            # Batch-convert all boxes at once and materialise as Python lists
            xywh_list = sv.xyxy_to_xywh(detections.xyxy).tolist()
            class_ids = detections.class_id.tolist()
            has_mask = detections.mask is not None

            img_anns: list[dict[str, Any]] = []
            for i in range(n_det):
                bx, by, bw, bh = xywh_list[i]
                cat_id = class_ids[i]

                ann: dict[str, Any] = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [bx, by, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }

                if has_mask:
                    ann["segmentation"] = []

                annotations.append(ann)
                anns[ann_id] = ann
                img_anns.append(ann)
                cat_to_imgs_set[cat_id].add(img_id)
                ann_id += 1

            img_to_anns[img_id] = img_anns

        self.dataset: dict[str, Any] = {
            "info": {"description": "RF-DETR YOLO dataset"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        self.imgs = imgs
        self.anns = anns
        self.cats: dict[int, dict[str, Any]] = {cat["id"]: cat for cat in categories}
        self.imgToAnns: dict[int, list] = img_to_anns
        self.catToImgs: dict[int, list] = {cat_id: list(img_set) for cat_id, img_set in cat_to_imgs_set.items()}

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """Get annotation IDs that satisfy given filter conditions.

        Args:
            imgIds: Filter by image IDs (list or single ID)
            catIds: Filter by category IDs (list or single ID)
            areaRng: Filter by area range [min, max]
            iscrowd: Filter by iscrowd flag (0 or 1)

        Returns:
            List of annotation IDs matching the filter conditions
        """
        imgIds = imgIds or []
        catIds = catIds or []
        areaRng = areaRng or []

        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        catIds = catIds if isinstance(catIds, list) else [catIds]

        if len(imgIds) == 0:
            anns = self.dataset["annotations"]
        else:
            anns = []
            for img_id in imgIds:
                anns.extend(self.imgToAnns.get(img_id, []))

        if len(catIds) > 0:
            anns = [ann for ann in anns if ann["category_id"] in catIds]

        if len(areaRng) == 2:
            anns = [ann for ann in anns if ann["area"] >= areaRng[0] and ann["area"] <= areaRng[1]]

        if iscrowd is not None:
            anns = [ann for ann in anns if ann["iscrowd"] == iscrowd]

        return [ann["id"] for ann in anns]

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """Get category IDs that satisfy given filter conditions.

        Args:
            catNms: Filter by category names (list)
            supNms: Filter by supercategory names (list, not used)
            catIds: Filter by category IDs (list)

        Returns:
            List of category IDs matching the filter conditions
        """
        catNms = catNms or []
        # supNms = supNms or []
        catIds = catIds or []

        cats = self.dataset["categories"]

        if len(catNms) > 0:
            cats = [cat for cat in cats if cat["name"] in catNms]
        if len(catIds) > 0:
            cats = [cat for cat in cats if cat["id"] in catIds]

        return [cat["id"] for cat in cats]

    def getImgIds(self, imgIds=None, catIds=None):
        """Get image IDs that satisfy given filter conditions.

        Args:
            imgIds: Filter to these image IDs (list)
            catIds: Filter by images containing these category IDs (list)

        Returns:
            List of image IDs matching the filter conditions
        """
        imgIds = imgIds or []
        catIds = catIds or []
        imgIds = set(imgIds) if imgIds else set(self.imgs.keys())

        if len(catIds) > 0:
            # Find all images that contain at least one of the specified categories
            matching_img_ids = set()
            for cat_id in catIds:
                matching_img_ids.update(self.catToImgs.get(cat_id, []))

            # Intersect with existing imgIds filter
            imgIds &= matching_img_ids

        return list(imgIds)

    def loadAnns(self, ids=None):
        """Load annotations with the specified IDs.

        Args:
            ids: Annotation IDs to load (list or single ID)

        Returns:
            List of annotation dicts with keys: id, image_id, category_id, bbox, area, iscrowd
        """
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.anns[ann_id] for ann_id in ids if ann_id in self.anns]

    def loadCats(self, ids=None):
        """Load categories with the specified IDs.

        Args:
            ids: Category IDs to load (list or single ID). If None, returns all categories.

        Returns:
            List of category dicts with keys: id, name, supercategory
        """
        if ids is None:
            return list(self.cats.values())
        ids = ids if isinstance(ids, list) else [ids]
        return [self.cats[cat_id] for cat_id in ids if cat_id in self.cats]

    def loadImgs(self, ids=None):
        """Load images with the specified IDs.

        Args:
            ids: Image IDs to load (list or single ID)

        Returns:
            List of image dicts with keys: id, file_name, height, width
        """
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.imgs[img_id] for img_id in ids if img_id in self.imgs]


class YoloDetection(VisionDataset):
    """YOLO format dataset with ``.cache`` file support for fast loading.

    On the first run the dataset scans all images and labels, then writes a
    ``.cache`` pickle next to the labels directory.  Subsequent
    instantiations with the same data load from cache almost instantly.

    Invalid label lines (non-numeric class IDs, malformed values) are
    automatically skipped and a summary warning is emitted.

    This class provides a VisionDataset interface compatible with RF-DETR
    training, matching the API of CocoDetection.

    Args:
        img_folder: Path to the directory containing images.
        lb_folder: Path to the directory containing YOLO annotation ``.txt``
            files.
        data_file: Path to ``data.yaml`` file containing class names.
        transforms: Optional transforms to apply to images and targets.
        include_masks: Whether to load segmentation masks (for YOLO
            segmentation format).
    """

    def __init__(
        self,
        img_folder: str,
        lb_folder: str,
        data_file: str,
        transforms: Any = None,
        include_masks: bool = False,
    ):
        super(YoloDetection, self).__init__(img_folder)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertYolo(include_masks=include_masks, normalized_coords=not include_masks)

        logger.info("Loading YOLO annotations from %s …", img_folder)
        self.sv_dataset, self._image_sizes = load_yolo_annotations_cached(
            images_directory_path=img_folder,
            annotations_directory_path=lb_folder,
            data_yaml_path=data_file,
            force_masks=include_masks,
        )

        self.classes = self.sv_dataset.classes
        self.ids = list(range(len(self.sv_dataset)))
        self._debug_first_batch = os.getenv("RFDETR_DEBUG_FIRST_BATCH", "0").lower() in {"1", "true", "yes", "on"}
        self._debug_trace_samples = int(os.getenv("RFDETR_DEBUG_TRACE_SAMPLES", "6"))
        self._debug_seen_samples = 0

        logger.info("Building COCO-compatible index for %d images …", len(self.ids))
        t0 = time.perf_counter()
        self.coco = CocoLikeAPI(self.classes, self.sv_dataset, self._image_sizes)
        logger.info("COCO-compatible index built in %.2fs", time.perf_counter() - t0)

    def __len__(self) -> int:
        return len(self.sv_dataset)

    def __getitem__(self, idx: int):
        """Load a single sample, skipping corrupt images gracefully.

        If an image cannot be read (e.g. truncated file on disk), the method
        retries with a random replacement index.  This mirrors the behaviour
        of Ultralytics YOLO, ensuring that a handful of broken files on a
        mechanical drive never crash a long training run.

        Args:
            idx: Sample index.

        Returns:
            A ``(image, target)`` tuple suitable for the training pipeline.
        """
        import random

        debug_first_batch = getattr(
            self,
            "_debug_first_batch",
            os.getenv("RFDETR_DEBUG_FIRST_BATCH", "0").lower() in {"1", "true", "yes", "on"},
        )
        debug_trace_samples = getattr(self, "_debug_trace_samples", int(os.getenv("RFDETR_DEBUG_TRACE_SAMPLES", "6")))
        debug_seen_samples = getattr(self, "_debug_seen_samples", 0)
        trace_this_sample = debug_first_batch and debug_seen_samples < debug_trace_samples
        rank = get_rank() if debug_first_batch else 0
        original_idx = idx
        sample_start = time.perf_counter() if trace_this_sample else 0.0
        if trace_this_sample:
            logger.error("[RFDETR-DEBUG][rank=%d] yolo __getitem__ start idx=%d", rank, idx)
            sys.stderr.write(f"[RFDETR-DEBUG][rank={rank}] yolo __getitem__ start idx={idx}\n")
            sys.stderr.flush()
        for attempt in range(10):
            try:
                image_id = self.ids[idx]
                image_path, cv2_image, detections = self.sv_dataset[idx]

                if cv2_image is None:
                    raise ValueError(f"cv2.imread returned None for {image_path}")

                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_image = cv2_image[:, :, ::-1]
                img = Image.fromarray(rgb_image)
                image_width, image_height = img.size
                if image_width <= 0 or image_height <= 0:
                    raise ValueError(f"Invalid image size for idx={idx}: width={image_width}, height={image_height}")

                target = {"image_id": image_id, "detections": detections}
                img, target = self.prepare(img, target)

                if self._transforms is not None:
                    img, target = self._transforms(img, target)

                if trace_this_sample:
                    elapsed = time.perf_counter() - sample_start
                    logger.error(
                        "[RFDETR-DEBUG][rank=%d] yolo __getitem__ ok idx=%d orig_idx=%d image_id=%d elapsed=%.3fs",
                        rank,
                        idx,
                        original_idx,
                        image_id,
                        elapsed,
                    )
                    sys.stderr.write(
                        f"[RFDETR-DEBUG][rank={rank}] yolo __getitem__ ok idx={idx} "
                        f"orig_idx={original_idx} image_id={image_id} elapsed={elapsed:.3f}s\n"
                    )
                    sys.stderr.flush()
                    self._debug_seen_samples = debug_seen_samples + 1

                return img, target
            except Exception as exc:
                logger.warning(
                    "Skipping corrupt image idx=%d, retrying with random replacement",
                    idx,
                )
                if debug_first_batch:
                    logger.error(
                        "[RFDETR-DEBUG][rank=%d] yolo retry attempt=%d/10 idx=%d err=%s: %s",
                        rank,
                        attempt + 1,
                        idx,
                        type(exc).__name__,
                        exc,
                    )
                    sys.stderr.write(
                        f"[RFDETR-DEBUG][rank={rank}] yolo retry attempt={attempt + 1}/10 "
                        f"idx={idx} err={type(exc).__name__}: {exc}\n"
                    )
                    sys.stderr.flush()
                idx = random.randint(0, len(self.sv_dataset) - 1)

        raise RuntimeError("Failed to load a valid sample after 10 attempts")


def build_roboflow_from_yolo(image_set: str, args: Any, resolution: int) -> YoloDetection:
    """Build a Roboflow YOLO-format dataset.

    This uses Roboflow's standard YOLO directory structure
    (train/valid/test folders with images/ and labels/ subdirectories).

    Args:
        image_set: Dataset split to load. One of ``"train"``, ``"val"``, or
            ``"test"``.
        args: Argument namespace. The following attributes are consumed:
            ``dataset_dir``, ``square_resize_div_64``, ``aug_config``,
            ``segmentation_head``, ``multi_scale``, ``expanded_scales``,
            ``do_random_resize_via_padding``, ``patch_size``, ``num_windows``.
            ``aug_config`` is forwarded to the transform builder; when
            ``None`` the builder falls back to the default
            :data:`~rfdetr.datasets.aug_config.AUG_CONFIG`.
        resolution: Target square resolution in pixels.

    Returns:
        A :class:`YoloDetection` dataset instance ready for use with a
        DataLoader.
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f"provided Roboflow path {root} does not exist"

    # YOLO format uses images/ and labels/ subdirectories
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "valid" / "images", root / "valid" / "labels"),
        "test": (root / "test" / "images", root / "test" / "labels"),
    }

    # Prefer data.yaml; fall back to data.yml if present; default to data.yaml for error reporting
    data_file = next((root / f for f in REQUIRED_YOLO_YAML_FILES if (root / f).exists()), root / "data.yaml")
    img_folder, lb_folder = PATHS[image_set.split("_")[0]]
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", None)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", None)
    num_windows = getattr(args, "num_windows", None)
    aug_config = getattr(args, "aug_config", None)

    logger.info("Building '%s' dataset from %s …", image_set, img_folder)
    t0 = time.perf_counter()

    if square_resize_div_64:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                aug_config=aug_config,
            ),
            include_masks=include_masks,
        )
    else:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                aug_config=aug_config,
            ),
            include_masks=include_masks,
        )

    logger.info(
        "Dataset '%s' ready: %d samples in %.2fs",
        image_set,
        len(dataset),
        time.perf_counter() - t0,
    )
    return dataset
