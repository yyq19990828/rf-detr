"""YOLO 标注缓存加速模块。

类似 Ultralytics 的 .cache 机制，首次加载时扫描所有图片和标注，
后续加载直接从 pickle 缓存中恢复，大幅加速大规模数据集的启动时间。
"""

import gc
import hashlib
import os
import pickle
import sys
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Tuple

import numpy as np
import supervision as sv
from PIL import Image

from rfdetr.utilities.logger import get_logger

logger = get_logger()

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

