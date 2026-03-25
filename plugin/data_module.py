"""自定义 DataModule：替换上游数据集为增强版本，添加 EDA、test fallback 等。"""

from pathlib import Path
from typing import Any, List, Optional, Union

import torch

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets import build_dataset
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class PluginDataModule(RFDETRDataModule):
    """覆盖 setup() 添加 EDA 支持和 test split fallback。"""

    def setup(self, stage: str = "") -> None:
        resolution = self.model_config.resolution
        ns = _namespace_from_configs(self.model_config, self.train_config)
        if stage == "fit":
            if self._dataset_train is None:
                self._dataset_train = build_dataset("train", ns, resolution)
                self._run_eda_if_needed(ns)
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)
        elif stage == "validate":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)
        elif stage == "test":
            if self._dataset_test is None:
                split = "test" if self.train_config.dataset_file == "roboflow" else "val"
                try:
                    self._dataset_test = build_dataset(split, ns, resolution)
                except (FileNotFoundError, AssertionError):
                    if split == "test":
                        logger.warning("Test split not found, falling back to validation split for testing.")
                        self._dataset_test = build_dataset("val", ns, resolution)
                    else:
                        raise
        elif stage == "predict":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)

    def _run_eda_if_needed(self, ns: Any) -> None:
        """训练前运行 EDA（仅 rank 0）。"""
        if not getattr(self.train_config, "run_eda", False):
            return

        from rfdetr.utilities.distributed import get_rank

        if get_rank() != 0:
            return

        from rfdetr.datasets.eda import run_eda

        def _run_eda_if_supported(dataset: torch.utils.data.Dataset, split_name: str) -> None:
            if getattr(dataset, "coco", None) is None:
                logger.warning("Skipping EDA for split '%s': dataset has no COCO API.", split_name)
                return
            run_eda(dataset, self.train_config.output_dir, split=split_name)

        dataset_dirs = ns.dataset_dir if isinstance(ns.dataset_dir, list) else [ns.dataset_dir]
        train_subdatasets = getattr(self._dataset_train, "datasets", None)
        if train_subdatasets is not None and len(dataset_dirs) == len(train_subdatasets):
            for index, (dataset_dir, train_subdataset) in enumerate(
                zip(dataset_dirs, train_subdatasets),
                start=1,
            ):
                _run_eda_if_supported(
                    train_subdataset,
                    f"train_dataset_{index}_{Path(dataset_dir).name}",
                )
        else:
            _run_eda_if_supported(self._dataset_train, "train")
