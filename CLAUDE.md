# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- Imports AGENTS.md, which contains agent roles, behavioral rules, and coding constraints for this project. -->

@AGENTS.md

---

# 项目补充说明

## 项目概述

RF-DETR 是由 Roboflow 开发的实时目标检测与实例分割模型，基于 DINOv2 + Transformer 架构（LW-DETR 变体）。在 COCO 基准上达到 SOTA 实时检测性能。当前版本 1.5.0。

## 自定义工具脚本（tools/）

这些是本地开发使用的自定义脚本，不属于上游仓库：

```bash
# 训练（支持模型选择、参数配置）
python tools/train.py --model medium --epochs 100 --batch-size 4

# 多 GPU 分布式训练
./tools/train.sh

# YOLO 格式数据集转 COCO 格式
python tools/yolo_to_coco_rfdetr_converter.py

# 导出 ONNX
python tools/export_base.py      # Base 模型导出
python tools/export_medium.py    # Medium 模型导出
```

`tools/train.py` 使用模型工厂模式，支持通过 `--model` 参数选择 nano/small/medium/large/base，最终调用 `model.train(**kwargs)`。

## 训练管线架构

训练流程跨多个文件，理解调用链很重要：

```
model.train(**kwargs)                    # detr.py: RFDETR.train()
  → RFDETR.get_train_config()            # detr.py: 创建 TrainConfig/SegmentationTrainConfig
  → RFDETR.train_from_config(config)     # detr.py: 设置 metrics sinks, class_names
    → Model.train()                      # main.py: 核心训练逻辑 (1200+ 行)
      ├─ populate_args() → argparse namespace
      ├─ build_criterion_and_postprocessors()  # models/lwdetr.py
      ├─ build_dataset("train"/"val")          # datasets/
      ├─ AdamW optimizer + LambdaLR scheduler
      ├─ ModelEma (if use_ema=True)
      └─ 训练循环:
          ├─ engine.train_one_epoch()    # engine.py: AMP + 梯度累积 + 多尺度
          ├─ engine.evaluate()           # engine.py: COCO 指标评估
          ├─ 回调: on_fit_epoch_end()    # metrics 日志记录
          └─ 早停检查 + checkpoint 保存
```

关键文件：
- `src/rfdetr/detr.py` — 所有模型类定义，`train()` 入口
- `src/rfdetr/main.py` — `Model` 类，完整训练逻辑
- `src/rfdetr/engine.py` — `train_one_epoch()` 和 `evaluate()`
- `src/rfdetr/config.py` — 所有配置类（Pydantic），`ModelConfig` + `TrainConfig`
- `src/rfdetr/models/lwdetr.py` — 核心模型构建（`build_model`, `build_criterion_and_postprocessors`）

## 模型变体

### 检测模型（detr.py）
| 类名 | 骨干网络 | 分辨率 | 备注 |
|------|---------|--------|------|
| `RFDETRNano` | dinov2_windowed_small | 384 | |
| `RFDETRSmall` | dinov2_windowed_small | 512 | |
| `RFDETRMedium` | dinov2_windowed_small | 576 | 推荐 |
| `RFDETRLarge` | dinov2_windowed_small | 704 | 替代已弃用版本 |
| `RFDETRBase` | dinov2_windowed_base | 560 | 已弃用 |
| `RFDETRXLarge` | - | 700 | 需要 rfdetr_plus，懒加载 |
| `RFDETR2XLarge` | - | 880 | 需要 rfdetr_plus，懒加载 |

### 分割模型（detr.py）
`RFDETRSegNano`、`RFDETRSegSmall`、`RFDETRSegMedium`、`RFDETRSegLarge`、`RFDETRSegXLarge`、`RFDETRSeg2XLarge` — 分辨率 312-768，均设置 `segmentation_head=True`。

### 模型层级关系
- `RFDETR`（基类 in detr.py）→ 各变体类
- `RFDETR.model` 是 `rfdetr.main.Model` 实例
- `RFDETR.model.model` 是底层 PyTorch 模块

## 配置体系（config.py）

`BaseConfig` → `ModelConfig` → 各模型专属 Config（如 `RFDETRMediumConfig`）

关键 ModelConfig 字段：
- `encoder`: 骨干类型（`dinov2_windowed_small` / `dinov2_windowed_base`）
- `hidden_dim`: 嵌入维度（256-384）
- `resolution`: 输入分辨率（需被 patch_size 整除）
- `num_queries` / `num_select`: 查询数量（100-300）
- `segmentation_head`: 是否启用分割头

关键 TrainConfig 字段：
- `lr` / `lr_encoder`: 学习率（默认 1e-4 / 1.5e-4）
- `batch_size` / `grad_accum_steps`: 批大小 / 梯度累积（默认 4/4）
- `use_ema` / `ema_decay`: EMA（默认 True / 0.993）
- `dataset_file`: 数据集类型（"coco" / "yolo" / "roboflow" / "o365"）
- `multi_scale`: 多尺度训练（默认 True）

## 数据集格式

支持 4 种数据集类型，通过 `dataset_file` 参数指定（`config.py` 中 `TrainConfig.dataset_file`）：

### 1. Roboflow 格式（默认，`dataset_file="roboflow"`）
自动检测目录结构是 COCO 还是 YOLO，无需手动指定。检测逻辑在 `datasets/__init__.py: detect_roboflow_format()`：
- 若 `train/_annotations.coco.json` 存在 → 按 COCO 处理
- 若 `data.yaml` + `train/images/` 存在 → 按 YOLO 处理

**重要：验证集文件夹命名约定** — 磁盘上的文件夹必须命名为 **`valid`**（不是 `val`）。代码内部 `build_dataset(image_set="val", ...)` 传入 `"val"`，在 PATHS 字典中映射到 `valid/` 目录（见 `coco.py:519`、`yolo.py:487`）。命名为 `val` 会导致路径不存在报错。

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
│   ├── images/            # 图片文件
│   └── labels/            # YOLO 格式 .txt 标注（class cx cy w h 归一化）
├── valid/                 # 注意：必须是 valid，不是 val
│   ├── images/
│   └── labels/
└── test/  (可选)
    ├── images/
    └── labels/
```
实现：`datasets/yolo.py` 中的 `YoloDetection` 类，底层使用 `supervision.DetectionDataset.from_yolo()` 加载，并通过 `CocoLikeAPI` 包装提供 COCO 兼容的评估接口。支持检测和分割（`include_masks=True`）。

**YOLO vs COCO 的类别 ID 差异**：YOLO 格式类别 ID 从 0 开始，`num_classes = len(class_names)`；Roboflow/COCO 格式为 `num_classes = len(class_names) + 1`（参见 `detr.py: train_from_config()`）。

### 4. Objects365（`dataset_file="o365"`）
大规模预训练数据集，通过 `datasets/o365.py` 实现。

### 手动转换工具
`tools/yolo_to_coco_rfdetr_converter.py` 可将 YOLO 格式离线转换为 COCO 格式，适用于需要标准 COCO JSON 的场景。但常规训练中直接使用 YOLO 格式即可（`dataset_file="yolo"` 或 `dataset_file="roboflow"` 自动识别）。

## Checkpoint 输出

训练产物保存在 `output_dir`（默认 `output/`）：
- `checkpoint_{epoch:04d}.pth` — 每轮 checkpoint
- `checkpoint_best.pth` — 最佳模型
- `checkpoint_best_ema.pth` — 最佳 EMA 模型

## 注意事项

- 永远使用中文回复用户
- 优先使用虚拟环境（`source .venv/bin/activate` 或 `uv` 管理）
- 每次测试完毕清理测试产生的中间产物
- `resolution` 必须能被 `patch_size`（通常 14）整除，即被 56 整除
- Plus 模型（XLarge/2XLarge）通过 `__init__.py` 中的 `__getattr__` 懒加载，需安装 `rfdetr_plus`
