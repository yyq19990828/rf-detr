# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""自定义模型：通过子类化注入自定义 DataModule 和 Trainer callbacks。"""

from typing import List, Union

from rfdetr.detr import RFDETR
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class PluginRFDETR(RFDETR):
    """通过覆盖钩子方法注入自定义 DataModule 和 Callbacks。"""

    def _build_data_module(self, model_config, train_config):
        from plugin.data_module import PluginDataModule

        return PluginDataModule(model_config, train_config)

    def _build_trainer(self, train_config, model_config, **kwargs):
        from rfdetr.training import build_trainer

        trainer = build_trainer(train_config, model_config, **kwargs)

        # 注入 epoch loss logger callback
        from plugin.callbacks.epoch_logger import EpochLoggerCallback

        trainer.callbacks.append(EpochLoggerCallback())

        # 注入 val_visualizer callback
        if getattr(train_config, "save_val_predictions", False):
            from plugin.callbacks.val_visualizer import ValVisualizerCallback

            trainer.callbacks.append(
                ValVisualizerCallback(
                    output_dir=train_config.output_dir,
                    save_interval=getattr(train_config, "eval_interval", 1),
                    max_images=9,
                )
            )

        return trainer

    @staticmethod
    def _load_classes(dataset_dir: Union[str, List[str]]) -> List[str]:
        """多目录类名加载和一致性校验。"""
        if isinstance(dataset_dir, str):
            return RFDETR._load_classes(dataset_dir)

        dirs = list(dataset_dir)
        if not dirs:
            raise ValueError("dataset_dir list must not be empty")

        reference_classes = RFDETR._load_classes(dirs[0])

        for d in dirs[1:]:
            other_classes = RFDETR._load_classes(d)
            if other_classes != reference_classes:
                raise ValueError(
                    f"Class name mismatch across dataset directories.\n"
                    f"  Directory '{dirs[0]}' has classes: {reference_classes}\n"
                    f"  Directory '{d}' has classes: {other_classes}\n"
                    f"All dataset directories must contain the exact same class names."
                )

        return reference_classes


# 各模型变体的 Plugin 版本 — 通过 MRO，train() 来自 PluginRFDETR，config 来自原始变体
def _make_plugin_variant(name, base_cls):
    """动态创建 Plugin 模型变体类。"""
    return type(name, (PluginRFDETR, base_cls), {})


def create_plugin_model_factory():
    """创建使用 Plugin 增强的模型工厂。"""
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    from rfdetr.detr import RFDETRBase

    # RFDETRBase 在上游已 deprecated，并被 deprecate proxy 包装；继续动态多继承
    # 会触发 metaclass conflict。base 保留为兼容入口，其余常用尺寸仍使用 Plugin。
    return {
        "nano": _make_plugin_variant("PluginRFDETRNano", RFDETRNano),
        "small": _make_plugin_variant("PluginRFDETRSmall", RFDETRSmall),
        "medium": _make_plugin_variant("PluginRFDETRMedium", RFDETRMedium),
        "large": _make_plugin_variant("PluginRFDETRLarge", RFDETRLarge),
        "base": RFDETRBase,
    }
