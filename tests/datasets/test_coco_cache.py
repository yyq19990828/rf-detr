# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import pickle
from pathlib import Path

import pytest

from rfdetr.datasets.coco import _CACHE_VERSION, _hash_file, load_coco_cached


def _write_coco_annotation(path: Path, categories: list[dict], *, image_id: int = 1) -> None:
    data = {
        "images": [{"id": image_id, "file_name": "dummy.jpg", "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": image_id,
                "category_id": categories[0]["id"],
                "bbox": [1, 2, 3, 4],
                "area": 12,
                "iscrowd": 0,
            }
        ],
        "categories": categories,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def ann_file(tmp_path: Path) -> Path:
    path = tmp_path / "train" / "_annotations.coco.json"
    _write_coco_annotation(path, categories=[{"id": 1, "name": "cat", "supercategory": "none"}])
    return path


def test_load_coco_cached_creates_cache_file(ann_file: Path) -> None:
    cache_path = ann_file.parent / ".coco_cache"
    assert not cache_path.exists()

    coco = load_coco_cached(ann_file)

    assert cache_path.exists()
    assert coco.dataset["categories"][0]["name"] == "cat"


def test_load_coco_cached_reuses_existing_cache(ann_file: Path) -> None:
    cache_path = ann_file.parent / ".coco_cache"

    first = load_coco_cached(ann_file)
    first_mtime = cache_path.stat().st_mtime_ns
    second = load_coco_cached(ann_file)
    second_mtime = cache_path.stat().st_mtime_ns

    assert first.dataset == second.dataset
    assert first_mtime == second_mtime


def test_load_coco_cached_invalidates_stale_cache(ann_file: Path) -> None:
    cache_path = ann_file.parent / ".coco_cache"
    load_coco_cached(ann_file)
    old_hash = pickle.loads(cache_path.read_bytes())["hash"]

    _write_coco_annotation(
        ann_file,
        categories=[
            {"id": 1, "name": "cat", "supercategory": "none"},
            {"id": 2, "name": "dog", "supercategory": "none"},
        ],
    )

    coco = load_coco_cached(ann_file)
    new_cache = pickle.loads(cache_path.read_bytes())

    assert new_cache["hash"] != old_hash
    assert new_cache["hash"] == _hash_file(ann_file)
    assert len(coco.dataset["categories"]) == 2


def test_load_coco_cached_recovers_from_corrupt_cache(ann_file: Path) -> None:
    cache_path = ann_file.parent / ".coco_cache"
    load_coco_cached(ann_file)

    cache_path.write_bytes(b"not-a-pickle")

    coco = load_coco_cached(ann_file)

    assert coco.dataset["images"][0]["id"] == 1
    recovered_cache = pickle.loads(cache_path.read_bytes())
    assert recovered_cache["version"] == _CACHE_VERSION


def test_load_coco_cached_ddp_nonzero_rank_waits_for_rank0_cache(
    monkeypatch: pytest.MonkeyPatch, ann_file: Path
) -> None:
    cache_path = ann_file.parent / ".coco_cache"
    assert not cache_path.exists()

    monkeypatch.setattr("rfdetr.datasets.coco.is_dist_avail_and_initialized", lambda: True)
    monkeypatch.setattr("rfdetr.datasets.coco.get_rank", lambda: 1)

    sleep_calls = {"count": 0}

    def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            coco_dict = json.loads(ann_file.read_text(encoding="utf-8"))
            dataset_hash = _hash_file(ann_file)
            cache_data = {
                "version": _CACHE_VERSION,
                "hash": dataset_hash,
                "dataset": coco_dict,
                "anns": {1: coco_dict["annotations"][0]},
                "cats": {1: coco_dict["categories"][0]},
                "imgs": {1: coco_dict["images"][0]},
                "imgToAnns": {1: [coco_dict["annotations"][0]]},
                "catToImgs": {1: [1]},
            }
            cache_path.write_bytes(pickle.dumps(cache_data))

    monkeypatch.setattr("rfdetr.datasets.coco.time.sleep", _fake_sleep)

    coco = load_coco_cached(ann_file)

    assert sleep_calls["count"] >= 1
    assert coco.dataset["categories"][0]["name"] == "cat"
