# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for ``ValVisualizerCallback``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch

from rfdetr.lit.callbacks.val_visualizer import ValVisualizerCallback


def _make_trainer(*, global_rank: int = 0, current_epoch: int = 0) -> MagicMock:
    """Create a minimal trainer mock for callback hook tests.

    Args:
        global_rank: Distributed rank to expose on trainer.
        current_epoch: Current epoch index.

    Returns:
        MagicMock trainer with the required attributes.
    """
    trainer = MagicMock(name="trainer")
    trainer.global_rank = global_rank
    trainer.current_epoch = current_epoch
    return trainer


def _make_batch(num_images: int = 2) -> tuple[MagicMock, list[dict[str, torch.Tensor]]]:
    """Create a minimal validation batch with synthetic tensors.

    Args:
        num_images: Number of images/targets in the batch.

    Returns:
        Tuple of ``(samples, targets)`` similar to datamodule output.
    """
    samples = MagicMock(name="samples")
    samples.tensors = torch.rand(num_images, 3, 16, 16)
    targets: list[dict[str, torch.Tensor]] = []
    for _ in range(num_images):
        targets.append(
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "orig_size": torch.tensor([16, 16], dtype=torch.int64),
            }
        )
    return samples, targets


def _make_outputs(num_images: int = 2, with_predictions: bool = True) -> dict[str, list[dict[str, torch.Tensor]]]:
    """Create mock validation_step outputs.

    Args:
        num_images: Number of per-image prediction dicts.
        with_predictions: Whether to include one prediction per image.

    Returns:
        Dict matching ``RFDETRModule.validation_step`` output schema.
    """
    results: list[dict[str, torch.Tensor]] = []
    for _ in range(num_images):
        if with_predictions:
            results.append(
                {
                    "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.9], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                }
            )
        else:
            results.append(
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
            )
    return {"results": results}


def test_collects_images_from_validation_batches() -> None:
    """Callback accumulates examples from validation batches up to max_images."""
    callback = ValVisualizerCallback(output_dir=Path("unused"), save_interval=1, max_images=9)
    trainer = _make_trainer(global_rank=0)
    batch = _make_batch(num_images=5)

    callback.on_validation_batch_end(trainer, MagicMock(), _make_outputs(num_images=5), batch, batch_idx=0)
    callback.on_validation_batch_end(trainer, MagicMock(), _make_outputs(num_images=5), batch, batch_idx=1)

    assert len(callback._examples) == 9


def test_saves_grid_at_correct_intervals() -> None:
    """Callback saves only when current_epoch matches save_interval."""
    callback = ValVisualizerCallback(output_dir=Path("unused"), save_interval=2, max_images=9)
    callback._save_grid = MagicMock(name="save_grid")
    trainer = _make_trainer(global_rank=0, current_epoch=2)

    callback.on_validation_batch_end(trainer, MagicMock(), _make_outputs(num_images=2), _make_batch(num_images=2), 0)
    callback.on_validation_epoch_end(trainer, MagicMock())

    callback._save_grid.assert_called_once_with(epoch=2)


def test_skips_saving_on_non_rank_zero() -> None:
    """Callback does not save on non-zero distributed ranks."""
    callback = ValVisualizerCallback(output_dir=Path("unused"), save_interval=1, max_images=9)
    callback._save_grid = MagicMock(name="save_grid")
    trainer = _make_trainer(global_rank=1, current_epoch=1)

    callback.on_validation_batch_end(trainer, MagicMock(), _make_outputs(num_images=2), _make_batch(num_images=2), 0)
    callback.on_validation_epoch_end(trainer, MagicMock())

    callback._save_grid.assert_not_called()


def test_handles_empty_predictions_gracefully() -> None:
    """Callback tolerates batches where predicted boxes are empty."""
    callback = ValVisualizerCallback(output_dir=Path("unused"), save_interval=1, max_images=9)
    callback._save_grid = MagicMock(name="save_grid")
    trainer = _make_trainer(global_rank=0, current_epoch=1)

    callback.on_validation_batch_end(
        trainer,
        MagicMock(),
        _make_outputs(num_images=2, with_predictions=False),
        _make_batch(num_images=2),
        0,
    )
    callback.on_validation_epoch_end(trainer, MagicMock())

    callback._save_grid.assert_called_once_with(epoch=1)


def test_clears_buffer_after_saving() -> None:
    """Callback clears accumulated examples after epoch-end save."""
    callback = ValVisualizerCallback(output_dir=Path("unused"), save_interval=1, max_images=9)
    callback._save_grid = MagicMock(name="save_grid")
    trainer = _make_trainer(global_rank=0, current_epoch=1)

    callback.on_validation_batch_end(trainer, MagicMock(), _make_outputs(num_images=2), _make_batch(num_images=2), 0)
    assert len(callback._examples) > 0

    callback.on_validation_epoch_end(trainer, MagicMock())

    assert callback._examples == []
