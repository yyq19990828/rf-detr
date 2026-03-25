# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Validation visualization callback for RF-DETR Lightning training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning import Callback

from rfdetr.util.box_ops import box_cxcywh_to_xyxy
from rfdetr.util.logger import get_logger

logger = get_logger()


class ValVisualizerCallback(Callback):
    """Saves a 3x3 grid of validation predictions every N epochs.

    Args:
        output_dir: Directory to save visualization grids.
        save_interval: Save grid every N validation epochs.
        max_images: Maximum images per grid (default 9 for 3x3).
    """

    def __init__(self, output_dir: Path | str, save_interval: int = 1, max_images: int = 9) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.save_interval = max(1, int(save_interval))
        self.max_images = min(9, max(1, int(max_images)))
        self._examples: list[dict[str, np.ndarray]] = []

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Collect up to ``max_images`` examples from validation batches.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Validation step output dict.
            batch: Validation batch tuple ``(samples, targets)``.
            batch_idx: Batch index within the validation epoch.
        """
        del pl_module, batch_idx
        if int(getattr(trainer, "global_rank", 0)) != 0:
            return
        if len(self._examples) >= self.max_images:
            return

        samples, targets = batch
        images = samples.tensors
        results = outputs.get("results", []) if isinstance(outputs, dict) else []

        for image_tensor, target, prediction in zip(images, targets, results):
            if len(self._examples) >= self.max_images:
                break
            self._examples.append(self._collect_example(image_tensor, target, prediction))

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Optionally save a validation visualization grid and clear the buffer.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        del pl_module
        should_save = (
            int(getattr(trainer, "global_rank", 0)) == 0
            and bool(self._examples)
            and int(getattr(trainer, "current_epoch", 0)) % self.save_interval == 0
        )
        if should_save:
            self._save_grid(epoch=int(getattr(trainer, "current_epoch", 0)))
        self._examples.clear()

    def _collect_example(
        self,
        image_tensor: torch.Tensor,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
    ) -> dict[str, np.ndarray]:
        """Convert one sample/target/prediction triple into plot-ready arrays.

        Args:
            image_tensor: Normalized image tensor in ``(C, H, W)``.
            target: Target dict with normalized CxCyWH boxes.
            prediction: Prediction dict with postprocessed xyxy boxes.

        Returns:
            Plot-ready numpy arrays for image and boxes.
        """
        image = self._denormalize_image(image_tensor)
        image_h, image_w = image.shape[0], image.shape[1]

        gt_boxes = target.get("boxes", torch.empty((0, 4), device=image_tensor.device))
        if gt_boxes.numel() > 0:
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
            gt_xyxy = gt_xyxy * gt_boxes.new_tensor([image_w, image_h, image_w, image_h])
            gt_xyxy_np = gt_xyxy.detach().cpu().float().numpy().astype(np.float32)
        else:
            gt_xyxy_np = np.empty((0, 4), dtype=np.float32)

        pred_boxes = prediction.get("boxes", torch.empty((0, 4), device=image_tensor.device))
        pred_scores = prediction.get("scores", torch.empty((0,), device=image_tensor.device))
        pred_xyxy_np = self._rescale_prediction_boxes(pred_boxes, target, image_w=image_w, image_h=image_h)
        pred_scores_np = pred_scores.detach().cpu().float().numpy().astype(np.float32)

        return {
            "image": image,
            "gt_boxes": gt_xyxy_np,
            "pred_boxes": pred_xyxy_np,
            "pred_scores": pred_scores_np,
        }

    def _rescale_prediction_boxes(
        self,
        pred_boxes: torch.Tensor,
        target: dict[str, torch.Tensor],
        image_w: int,
        image_h: int,
    ) -> np.ndarray:
        """Rescale postprocessed prediction boxes to the plotted image size.

        Args:
            pred_boxes: Postprocessed xyxy boxes.
            target: Target dict containing ``orig_size`` when available.
            image_w: Plotted image width.
            image_h: Plotted image height.

        Returns:
            Prediction boxes in xyxy numpy format aligned to the plotted image.
        """
        if pred_boxes.numel() == 0:
            return np.empty((0, 4), dtype=np.float32)

        resized_boxes = pred_boxes.detach().clone()
        orig_size = target.get("orig_size")
        if isinstance(orig_size, torch.Tensor) and orig_size.numel() == 2:
            orig_h, orig_w = int(orig_size[0].item()), int(orig_size[1].item())
            if orig_h > 0 and orig_w > 0:
                resized_boxes[:, [0, 2]] = resized_boxes[:, [0, 2]] * (float(image_w) / float(orig_w))
                resized_boxes[:, [1, 3]] = resized_boxes[:, [1, 3]] * (float(image_h) / float(orig_h))

        return resized_boxes.cpu().float().numpy().astype(np.float32)

    def _denormalize_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Convert a normalized image tensor into a uint8 numpy image.

        Args:
            image_tensor: Normalized tensor in ``(C, H, W)``.

        Returns:
            Image array in ``(H, W, C)`` with uint8 dtype.
        """
        inv_mean = image_tensor.new_tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(3, 1, 1)
        inv_std = image_tensor.new_tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(3, 1, 1)
        denormalized = image_tensor * inv_std + inv_mean
        image_np = denormalized.detach().cpu().float().permute(1, 2, 0).numpy()
        return (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _save_grid(self, epoch: int) -> None:
        """Render and save a 3x3 validation grid with GT and predictions.

        Args:
            epoch: Current zero-based epoch index.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes_flat = axes.flatten()

        for idx, ax in enumerate(axes_flat):
            if idx >= len(self._examples):
                ax.axis("off")
                continue

            example = self._examples[idx]
            ax.imshow(example["image"])
            ax.axis("off")

            for x1, y1, x2, y2 in example["gt_boxes"]:
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="green",
                        linewidth=2.0,
                        linestyle="solid",
                    )
                )

            for box, score in zip(example["pred_boxes"], example["pred_scores"]):
                x1, y1, x2, y2 = box.tolist()
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2.0,
                        linestyle="dashed",
                    )
                )
                ax.text(
                    x1,
                    y1,
                    f"{float(score):.2f}",
                    color="red",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
                )

        fig.tight_layout()
        save_path = self.output_dir / f"val_predictions_epoch_{epoch:04d}.jpg"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        logger.info("Saved validation prediction grid to %s", save_path)
