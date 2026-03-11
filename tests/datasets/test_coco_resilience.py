# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Resilience tests for ``CocoDetection.__getitem__`` retry behavior."""

import pytest
import torchvision

from rfdetr.datasets import coco as coco_module
from rfdetr.datasets.coco import CocoDetection


def _build_dataset(ids: list[int]) -> CocoDetection:
    dataset = CocoDetection.__new__(CocoDetection)
    dataset.ids = ids
    dataset.prepare = lambda img, target: (img, target)
    dataset._transforms = None
    return dataset


def test_getitem_succeeds_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build_dataset(ids=[123])

    def mock_super_getitem(self: CocoDetection, idx: int):
        return f"img-{idx}", [{"id": idx}]

    monkeypatch.setattr(torchvision.datasets.CocoDetection, "__getitem__", mock_super_getitem)

    img, target = dataset[0]

    assert img == "img-0"
    assert target["image_id"] == 123
    assert target["annotations"] == [{"id": 0}]


def test_getitem_retries_with_random_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build_dataset(ids=[100, 101])

    seen_indices: list[int] = []

    def mock_super_getitem(self: CocoDetection, idx: int):
        seen_indices.append(idx)
        if idx == 0:
            raise OSError("corrupt image")
        return "ok-image", [{"id": idx}]

    monkeypatch.setattr(torchvision.datasets.CocoDetection, "__getitem__", mock_super_getitem)
    monkeypatch.setattr("random.randint", lambda _a, _b: 1)

    img, target = dataset[0]

    assert img == "ok-image"
    assert target["image_id"] == 101
    assert target["annotations"] == [{"id": 1}]
    assert seen_indices == [0, 1]


def test_getitem_raises_runtime_error_after_ten_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build_dataset(ids=[42])

    def mock_super_getitem(self: CocoDetection, idx: int):
        raise OSError(f"always corrupt: {idx}")

    monkeypatch.setattr(torchvision.datasets.CocoDetection, "__getitem__", mock_super_getitem)
    monkeypatch.setattr("random.randint", lambda _a, _b: 0)

    with pytest.raises(RuntimeError, match="Failed to load a valid sample after 10 attempts"):
        dataset[0]


def test_getitem_logs_warning_on_corrupt_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _build_dataset(ids=[7, 8])
    warnings: list[str] = []

    def mock_super_getitem(self: CocoDetection, idx: int):
        if idx == 0:
            raise ValueError("bad image")
        return "recovered", [{"id": idx}]

    monkeypatch.setattr(torchvision.datasets.CocoDetection, "__getitem__", mock_super_getitem)
    monkeypatch.setattr("random.randint", lambda _a, _b: 1)
    monkeypatch.setattr(coco_module.logger, "warning", lambda msg, idx: warnings.append(msg % idx))

    dataset[0]
    assert warnings == ["Skipping corrupt image idx=0, retrying with random replacement"]


def test_getitem_retries_on_invalid_zero_sized_image(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build_dataset(ids=[10, 11])
    seen_indices: list[int] = []

    class _MockImage:
        def __init__(self, width: int, height: int) -> None:
            self.size = (width, height)

    def mock_super_getitem(self: CocoDetection, idx: int):
        seen_indices.append(idx)
        if idx == 0:
            return _MockImage(0, 100), [{"id": idx}]
        return _MockImage(100, 100), [{"id": idx}]

    monkeypatch.setattr(torchvision.datasets.CocoDetection, "__getitem__", mock_super_getitem)
    monkeypatch.setattr("random.randint", lambda _a, _b: 1)

    image, target = dataset[0]

    assert image.size == (100, 100)
    assert target["image_id"] == 11
    assert seen_indices == [0, 1]
