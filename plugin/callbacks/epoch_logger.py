# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Callback：每个 epoch 结束时打印训练/验证 loss 摘要到终端。"""

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class EpochLoggerCallback(Callback):
    """每个 train/val epoch 结束时输出格式化的 loss 摘要。"""

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        loss = metrics.get("train/loss")
        loss_ce = metrics.get("train/loss_ce")
        loss_bbox = metrics.get("train/loss_bbox")
        loss_giou = metrics.get("train/loss_giou")
        class_error = metrics.get("train/class_error")
        lr = metrics.get("train/lr")

        parts = [f"Epoch {epoch:>3d}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if loss_ce is not None:
            parts.append(f"ce={loss_ce:.4f}")
        if loss_bbox is not None:
            parts.append(f"bbox={loss_bbox:.4f}")
        if loss_giou is not None:
            parts.append(f"giou={loss_giou:.4f}")
        if class_error is not None:
            parts.append(f"cls_err={class_error:.1f}%")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")

        logger.info("[Train] %s", "  ".join(parts))

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        val_loss = metrics.get("val/loss")
        mAP = metrics.get("val/mAP_50_95")
        mAP_50 = metrics.get("val/mAP_50")
        mAR = metrics.get("val/mAR")
        f1 = metrics.get("val/F1")
        ema_mAP = metrics.get("val/ema_mAP_50_95")

        parts = [f"Epoch {epoch:>3d}"]
        if val_loss is not None:
            parts.append(f"loss={val_loss:.4f}")
        if mAP is not None:
            parts.append(f"mAP={mAP:.4f}")
        if mAP_50 is not None:
            parts.append(f"mAP50={mAP_50:.4f}")
        if mAR is not None:
            parts.append(f"mAR={mAR:.4f}")
        if f1 is not None:
            parts.append(f"F1={f1:.4f}")
        if ema_mAP is not None:
            parts.append(f"ema_mAP={ema_mAP:.4f}")

        logger.info("[Val]   %s", "  ".join(parts))
