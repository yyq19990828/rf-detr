# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""增强版 YoloDetection：添加 corrupt resilience + debug tracing + annotation caching。"""

import os
import time

from plugin.datasets.resilience import with_corrupt_resilience, with_debug_tracing
from plugin.datasets.yolo_cache import load_yolo_annotations_cached
from rfdetr.datasets.yolo import CocoLikeAPI, ConvertYolo, YoloDetection
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class RobustYoloDetection(YoloDetection):
    """在上游 YoloDetection 基础上叠加缓存加载、corrupt image 重试和 debug tracing。"""

    def __init__(self, img_folder, lb_folder, data_file, transforms=None, include_masks=False):
        # 跳过上游 __init__ 中的 sv.DetectionDataset.from_yolo，使用缓存版本
        # 直接调用 VisionDataset.__init__
        from torchvision.datasets import VisionDataset

        VisionDataset.__init__(self, img_folder)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertYolo(include_masks=include_masks)

        logger.info("Loading YOLO annotations from %s …", img_folder)
        self.sv_dataset, self._image_sizes = load_yolo_annotations_cached(
            images_directory_path=img_folder,
            annotations_directory_path=lb_folder,
            data_yaml_path=data_file,
            force_masks=include_masks,
        )

        self.classes = self.sv_dataset.classes
        self.ids = list(range(len(self.sv_dataset)))

        logger.info("Building COCO-compatible index for %d images …", len(self.ids))
        t0 = time.perf_counter()
        self.coco = CocoLikeAPI(self.classes, self.sv_dataset, self._image_sizes)
        logger.info("COCO-compatible index built in %.2fs", time.perf_counter() - t0)

        self._debug_first_batch = os.getenv("RFDETR_DEBUG_FIRST_BATCH", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._debug_trace_samples = int(os.getenv("RFDETR_DEBUG_TRACE_SAMPLES", "6"))
        self._debug_seen_samples = 0

    @with_corrupt_resilience(max_retries=10)
    @with_debug_tracing
    def __getitem__(self, idx):
        return super().__getitem__(idx)
