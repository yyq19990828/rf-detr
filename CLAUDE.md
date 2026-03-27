# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- Imports AGENTS.md, which contains agent roles, behavioral rules, and coding constraints for this project. -->

@AGENTS.md

---

# 项目补充说明

## 项目概述

RF-DETR 是由 Roboflow 开发的实时目标检测与实例分割模型，基于 DINOv2 + Transformer 架构（LW-DETR 变体）。在 COCO 基准上达到 SOTA 实时检测性能。当前版本 1.6.0。

## 非侵入式 Plugin 架构

本项目采用 **plugin 架构**将自定义增强功能与上游代码解耦，上游更新时只需在 `plugin/` 中适配，`src/rfdetr/` 保持零修改（仅 `detr.py` 有 2 个钩子方法）。

### 架构总览

```
tools/train.py                         ← 入口，使用 plugin 模型工厂
    │
    ▼
plugin/                                ← 所有自定义代码
├── model.py                           PluginRFDETR（覆盖 _build_data_module + _build_trainer）
├── data_module.py                     PluginDataModule（EDA + test fallback）
├── datasets/
│   ├── resilience.py                  装饰器：corrupt retry + debug tracing
│   ├── robust_coco.py                 RobustCocoDetection（叠加装饰器）
│   ├── robust_yolo.py                 RobustYoloDetection（缓存 + 装饰器）
│   └── yolo_cache.py                  load_yolo_annotations_cached
└── callbacks/
    ├── val_visualizer.py              验证集预测可视化
    └── epoch_logger.py                Epoch 结束时打印 loss 摘要
    │
    ▼
src/rfdetr/  (上游代码，几乎零修改)     ← git merge 无冲突
└── detr.py 中仅有 2 个钩子方法:
    ├── _build_data_module()           可被子类覆盖以注入自定义 DataModule
    └── _build_trainer()               可被子类覆盖以注入自定义 Callbacks
```

### 调用链

```
tools/train.py
  → create_plugin_model_factory()       # plugin/model.py: 动态创建 Plugin 变体
  → PluginRFDETRNano().train(**kwargs)
    → RFDETR.train()                    # detr.py: 上游 train() 逻辑
      → self._build_data_module()       # 钩子 → plugin/data_module.py: PluginDataModule
      │   └─ setup(): build_dataset() + EDA + test fallback
      → self._build_trainer()           # 钩子 → plugin/model.py: 注入自定义 callbacks
      │   └─ EpochLoggerCallback + ValVisualizerCallback
      → trainer.fit(module, datamodule)
```

### 上游更新时的操作

1. `git fetch origin && git merge origin/HEAD` — `src/rfdetr/` 基本无冲突
2. 检查 `detr.py` 的 `train()` 方法签名是否变化（钩子调用点）
3. 在 `plugin/` 中适配上游接口变化（如 `build_dataset` 签名、Config 字段）

### Plugin 功能清单

| 功能                       | 文件                                 | 说明                                                            |
| -------------------------- | ------------------------------------ | --------------------------------------------------------------- |
| corrupt image resilience   | `plugin/datasets/resilience.py`      | `@with_corrupt_resilience` 装饰器，最多重试 10 次               |
| debug tracing              | `plugin/datasets/resilience.py`      | `@with_debug_tracing` 装饰器，`RFDETR_DEBUG_FIRST_BATCH=1` 启用 |
| YOLO annotation caching    | `plugin/datasets/yolo_cache.py`      | 类 Ultralytics 的 `.cache` 机制，加速大规模数据集启动           |
| EDA                        | `plugin/data_module.py`              | 训练前自动运行探索性数据分析                                    |
| multi-dir class validation | `plugin/model.py`                    | `_load_classes()` 多目录类名一致性校验                          |
| val visualizer             | `plugin/callbacks/val_visualizer.py` | 验证集 3x3 预测网格可视化                                       |
| epoch logger               | `plugin/callbacks/epoch_logger.py`   | 每 epoch 打印 loss/mAP 摘要                                     |
| test split fallback        | `plugin/data_module.py`              | test/ 不存在时自动回退到 val/                                   |

## 自定义工具脚本（tools/）

这些是本地开发使用的自定义脚本，不属于上游仓库：

```bash
# 训练（支持模型选择、参数配置，使用 plugin 增强）
python tools/train.py --model medium --dataset-dir datasets/custom --epochs 100 --batch-size 4

# 多目录联合训练
python tools/train.py --model medium --dataset-dir datasets/a --dataset-dir datasets/b --epochs 100

# 多 GPU 分布式训练
./tools/train.sh

# 导出 ONNX
python tools/export_base.py      # Base 模型导出
python tools/export_medium.py    # Medium 模型导出
python tools/export_small.py     # Small 模型导出
```

`tools/train.py` 通过 `plugin/model.py` 的 `create_plugin_model_factory()` 创建增强版模型，支持通过 `--model` 参数选择 nano/small/medium/large/base。

## 训练管线架构

训练基于 PyTorch Lightning，调用链：

```
model.train(**kwargs)                           # detr.py: RFDETR.train()
  → RFDETR.get_train_config()                   # detr.py: 创建 TrainConfig
  → self._build_data_module()                   # 钩子: PluginDataModule
  │   └─ setup("fit"):
  │       ├─ build_dataset("train"/"val")       # datasets/__init__.py
  │       └─ _run_eda_if_needed()               # plugin 自定义
  → self._build_trainer()                       # 钩子: 注入自定义 callbacks
  │   ├─ build_trainer()                        # training/trainer.py: PTL Trainer
  │   ├─ EpochLoggerCallback                    # plugin 自定义
  │   └─ ValVisualizerCallback                  # plugin 自定义
  → trainer.fit(module, datamodule)             # PyTorch Lightning 训练循环
      ├─ RFDETRModelModule.training_step()      # training/module_model.py
      ├─ COCOEvalCallback                       # training/callbacks/coco_eval.py
      ├─ BestModelCallback + EarlyStopping      # training/callbacks/best_model.py
      └─ RFDETREMACallback                      # training/callbacks/ema.py
```

关键文件：

- `src/rfdetr/detr.py` — 模型基类，`train()` 入口 + 钩子方法
- `src/rfdetr/training/trainer.py` — `build_trainer()`，PTL Trainer 构建
- `src/rfdetr/training/module_model.py` — `RFDETRModelModule`，PTL LightningModule
- `src/rfdetr/training/module_data.py` — `RFDETRDataModule`，PTL LightningDataModule
- `src/rfdetr/config.py` — 所有配置类（Pydantic），`ModelConfig` + `TrainConfig`
- `src/rfdetr/models/lwdetr.py` — 核心模型构建

## 模型变体

### 检测模型（detr.py）

| 类名            | 骨干网络              | 分辨率 | 备注                     |
| --------------- | --------------------- | ------ | ------------------------ |
| `RFDETRNano`    | dinov2_windowed_small | 384    |                          |
| `RFDETRSmall`   | dinov2_windowed_small | 512    |                          |
| `RFDETRMedium`  | dinov2_windowed_small | 576    | 推荐                     |
| `RFDETRLarge`   | dinov2_windowed_small | 704    | 替代已弃用版本           |
| `RFDETRBase`    | dinov2_windowed_base  | 560    | 已弃用                   |
| `RFDETRXLarge`  | -                     | 700    | 需要 rfdetr_plus，懒加载 |
| `RFDETR2XLarge` | -                     | 880    | 需要 rfdetr_plus，懒加载 |

### 分割模型（detr.py）

`RFDETRSegNano`、`RFDETRSegSmall`、`RFDETRSegMedium`、`RFDETRSegLarge`、`RFDETRSegXLarge`、`RFDETRSeg2XLarge` — 分辨率 312-768，均设置 `segmentation_head=True`。

### 模型层级关系

- `RFDETR`（基类 in detr.py）→ 各变体类
- `PluginRFDETR`（plugin/model.py）→ 通过 MRO 覆盖 `_build_data_module` / `_build_trainer`
- `RFDETR.model` 是 `rfdetr.inference.ModelContext` 实例
- `RFDETR.model.model` 是底层 PyTorch 模块

## 配置体系（config.py）

`BaseModel` → `ModelConfig` → 各模型专属 Config（如 `RFDETRMediumConfig`）

关键 ModelConfig 字段：

- `encoder`: 骨干类型（`dinov2_windowed_small` / `dinov2_windowed_base`）
- `hidden_dim`: 嵌入维度（256-384）
- `resolution`: 输入分辨率（需被 patch_size 整除）
- `num_queries` / `num_select`: 查询数量（100-300）
- `segmentation_head`: 是否启用分割头

关键 TrainConfig 字段：

- `lr` / `lr_encoder`: 学习率（默认 1e-4 / 1.5e-4）
- `batch_size` / `grad_accum_steps`: 批大小 / 梯度累积（默认 4/4，支持 `"auto"`）
- `use_ema` / `ema_decay`: EMA（默认 True / 0.993）
- `dataset_file`: 数据集类型（"coco" / "yolo" / "roboflow" / "o365"）
- `dataset_dir`: 数据集路径（支持 `Union[str, List[str]]` 多目录）
- `multi_scale`: 多尺度训练（默认 True）
- `progress_bar`: 进度条样式（`"tqdm"` / `"rich"` / `None`）
- `run_eda`: 训练前运行 EDA（默认 True）
- `save_val_predictions`: 保存验证集预测可视化（默认 True）

## 数据集格式

支持 4 种数据集类型，通过 `dataset_file` 参数指定：

### 1. Roboflow 格式（默认，`dataset_file="roboflow"`）

自动检测目录结构是 COCO 还是 YOLO，无需手动指定。

### 2. COCO 格式（`dataset_file="coco"`）

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg / *.png
├── valid/                 # 注意：必须是 valid，不是 val
│   ├── _annotations.coco.json
│   └── *.jpg / *.png
└── test/  (可选)
    ├── _annotations.coco.json
    └── *.jpg / *.png
```

### 3. YOLO 格式（`dataset_file="yolo"`，原生支持，无需转换）

```
dataset/
├── data.yaml              # 必须，包含类别名称 (names) 等信息
├── train/
│   ├── images/
│   └── labels/            # YOLO 格式 .txt 标注（class cx cy w h 归一化）
├── valid/                 # 注意：必须是 valid，不是 val
│   ├── images/
│   └── labels/
└── test/  (可选)
    ├── images/
    └── labels/
```

### 4. Objects365（`dataset_file="o365"`）

大规模预训练数据集。

## Checkpoint 输出

训练产物保存在 `output_dir`（默认 `output/`）：

- `checkpoint_{epoch}.ckpt` — 间隔 checkpoint（PTL 格式）
- `last.ckpt` — 最新 checkpoint
- `checkpoint_best_regular.pth` — 最佳常规模型
- `checkpoint_best_ema.pth` — 最佳 EMA 模型
- `val_predictions_epoch_NNNN.jpg` — 验证集预测可视化（plugin 提供）
- `metrics.csv` — 训练/验证指标日志

## 注意事项

- 永远使用中文回复用户
- 优先使用虚拟环境（`uv` 管理，`uv sync --all-groups`）
- 每次测试完毕清理测试产生的中间产物
- `resolution` 必须能被 `patch_size`（通常 14）整除，即被 56 整除
- Plus 模型（XLarge/2XLarge）通过 `__init__.py` 中的 `__getattr__` 懒加载，需安装 `rfdetr_plus`
- **修改自定义功能时，只改 `plugin/` 目录，不要修改 `src/rfdetr/` 中的上游代码**
- `transformers>=5.1.0` 是必须依赖（上游 v1.6.0 要求）
