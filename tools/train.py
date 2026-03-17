#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
RF-DETR 训练脚本

使用命令行参数进行模型训练的优化脚本。
支持多种模型变体和与当前 `model.train()` 对齐的训练参数配置。

使用示例:
    python tools/train.py --model medium --dataset-dir datasets/coco --epochs 100
    python tools/train.py --model nano --dataset-dir datasets/custom --batch-size 8 --lr 2e-4
    python tools/train.py --model medium --dataset-dir datasets/a --dataset-dir datasets/b --epochs 100
    python tools/train.py --model small --dataset-dir datasets/custom --resume output/checkpoint.pth --eval
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


def create_model_factory():
    """
    创建模型工厂函数，根据模型名称返回相应的模型实例。

    Returns:
        Dict[str, callable]: 模型名称到模型类的映射
    """
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    from rfdetr.detr import RFDETRBase

    return {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge, "base": RFDETRBase}


def add_boolean_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: Optional[bool],
    enable_help: str,
    disable_help: Optional[str] = None,
) -> None:
    """
    添加支持 `--foo` / `--no-foo` 的布尔参数。

    Args:
        parser: 参数解析器。
        name: 参数名，使用下划线形式。
        default: 默认值，允许为 None 以保留三态配置。
        enable_help: 启用该选项时的帮助信息。
        disable_help: 禁用该选项时的帮助信息。
    """
    option = name.replace("_", "-")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{option}", dest=name, action="store_true", help=enable_help)
    group.add_argument(
        f"--no-{option}",
        dest=name,
        action="store_false",
        help=disable_help or f"禁用{enable_help.removeprefix('启用')}",
    )
    parser.set_defaults(**{name: default})


def parse_json_argument(value: str) -> Dict[str, Any]:
    """
    解析 JSON 字符串参数。

    Args:
        value: JSON 字符串。

    Returns:
        Dict[str, Any]: 解析后的字典。

    Raises:
        argparse.ArgumentTypeError: 当输入不是合法 JSON 对象时抛出。
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"无效的 JSON: {error}") from error

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError('参数必须是 JSON 对象，例如 \'{"HorizontalFlip": {"p": 0.5}}\'')

    return parsed


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        argv: 可选的命令行参数列表，默认读取 sys.argv。

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="RF-DETR 模型训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python tools/train.py --model medium --dataset-dir datasets/coco --epochs 100
  python tools/train.py --model nano --dataset-dir datasets/custom --dataset-file yolo --batch-size 8 --lr 2e-4
  python tools/train.py --model medium --dataset-dir datasets/a --dataset-dir datasets/b --epochs 100 --resume output/checkpoint.pth
  python tools/train.py --model small --dataset-dir datasets/custom --dataset-file yolo --resume output/checkpoint.pth --eval
  python tools/train.py --model small --dataset-dir datasets/custom --wandb --project my-project --run exp-01
        """,
    )

    # 模型选择
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["nano", "small", "medium", "large", "base"],
        default="medium",
        help="选择模型变体 (默认: medium)",
    )

    # 数据集参数
    parser.add_argument(
        "--dataset-dir",
        type=str,
        action="append",
        required=True,
        help="数据集目录路径。可重复传入实现多目录联合训练，例如 --dataset-dir d1 --dataset-dir d2",
    )

    parser.add_argument(
        "--dataset-file",
        type=str,
        choices=["coco", "roboflow", "yolo"],
        default="roboflow",
        help="数据集格式 (默认: roboflow)",
    )

    # 训练基础参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认: 100)")

    parser.add_argument("--batch-size", type=int, default=4, help="批大小 (默认: 4)")

    parser.add_argument("--grad-accum-steps", type=int, default=4, help="梯度累积步数 (默认: 4)")

    parser.add_argument("--resume", type=str, default=None, help="从指定检查点恢复训练")

    parser.add_argument("--eval", action="store_true", default=False, help="仅执行验证评估，不进入训练循环")

    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认不显式设置")

    # 学习率参数
    parser.add_argument("--lr", type=float, default=1e-4, help="主学习率 (默认: 1e-4)")

    parser.add_argument("--lr-encoder", type=float, default=1.5e-4, help="编码器学习率 (默认: 1.5e-4)")

    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减 (默认: 1e-4)")

    parser.add_argument("--lr-drop", type=int, default=100, help="学习率衰减轮数 (默认: 100)")

    parser.add_argument("--warmup-epochs", type=float, default=0.0, help="学习率预热轮数 (默认: 0.0)")

    parser.add_argument("--lr-vit-layer-decay", type=float, default=0.8, help="ViT 分层学习率衰减 (默认: 0.8)")

    parser.add_argument("--lr-component-decay", type=float, default=0.7, help="组件学习率衰减 (默认: 0.7)")

    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["step", "cosine"],
        default="step",
        help="学习率调度器类型 (默认: step)",
    )

    parser.add_argument("--lr-min-factor", type=float, default=0.0, help="cosine 调度器的最小学习率比例 (默认: 0.0)")

    parser.add_argument("--clip-max-norm", type=float, default=0.1, help="梯度裁剪最大范数 (默认: 0.1)")

    parser.add_argument("--drop-path", type=float, default=0.0, help="DropPath 概率 (默认: 0.0)")

    # 输出目录
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录 (默认: output)")

    # EMA 参数
    add_boolean_argument(
        parser, "use_ema", default=False, enable_help="启用指数移动平均", disable_help="禁用指数移动平均"
    )

    parser.add_argument("--ema-decay", type=float, default=0.993, help="EMA衰减率 (默认: 0.993)")

    parser.add_argument("--ema-tau", type=int, default=100, help="EMA tau 参数 (默认: 100)")

    parser.add_argument("--ema-update-interval", type=int, default=1, help="EMA 更新间隔 (默认: 1)")

    # 早停参数
    parser.add_argument("--early-stopping", action="store_true", default=False, help="启用早停机制")

    parser.add_argument("--early-stopping-patience", type=int, default=10, help="早停耐心值 (默认: 10)")

    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001, help="早停最小改进量 (默认: 0.001)")

    add_boolean_argument(
        parser,
        "early_stopping_use_ema",
        default=False,
        enable_help="早停时基于 EMA 指标判断",
        disable_help="早停时不基于 EMA 指标判断",
    )

    # 日志记录
    add_boolean_argument(
        parser,
        "tensorboard",
        default=True,
        enable_help="启用 TensorBoard 日志",
        disable_help="禁用 TensorBoard 日志",
    )

    parser.add_argument("--wandb", action="store_true", default=False, help="启用Weights & Biases日志")

    parser.add_argument("--mlflow", action="store_true", default=False, help="启用 MLflow 日志")

    parser.add_argument("--clearml", action="store_true", default=False, help="启用 ClearML 日志")

    parser.add_argument("--project", type=str, default=None, help="W&B项目名称")

    parser.add_argument("--run", type=str, default=None, help="实验运行名称")

    # 数据处理参数
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载器工作进程数 (默认: 2)")

    parser.add_argument("--prefetch-factor", type=int, default=None, help="每个 worker 预取批次数，默认沿用框架设置")

    add_boolean_argument(
        parser,
        "pin_memory",
        default=None,
        enable_help="显式启用 DataLoader pin_memory",
        disable_help="显式禁用 DataLoader pin_memory",
    )

    add_boolean_argument(
        parser,
        "persistent_workers",
        default=None,
        enable_help="显式启用 DataLoader persistent_workers",
        disable_help="显式禁用 DataLoader persistent_workers",
    )

    add_boolean_argument(
        parser,
        "multi_scale",
        default=True,
        enable_help="启用多尺度训练",
        disable_help="禁用多尺度训练",
    )

    add_boolean_argument(
        parser,
        "expanded_scales",
        default=True,
        enable_help="启用扩展尺度采样",
        disable_help="禁用扩展尺度采样",
    )

    add_boolean_argument(
        parser,
        "square_resize_div_64",
        default=True,
        enable_help="启用 64 对齐的方形缩放",
        disable_help="禁用 64 对齐的方形缩放",
    )

    add_boolean_argument(
        parser,
        "do_random_resize_via_padding",
        default=False,
        enable_help="启用基于 padding 的随机缩放",
        disable_help="禁用基于 padding 的随机缩放",
    )

    # 进度条
    add_boolean_argument(
        parser,
        "progress_bar",
        default=False,
        enable_help="显示 tqdm 训练进度条",
        disable_help="禁用 tqdm 训练进度条",
    )

    # 其他训练参数
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="检查点保存间隔 (默认: 10)")

    parser.add_argument("--group-detr", type=int, default=13, help="Group-DETR 分组数 (默认: 13)")

    parser.add_argument("--num-select", type=int, default=300, help="后处理保留的查询数 (默认: 300)")

    parser.add_argument("--cls-loss-coef", type=float, default=1.0, help="分类损失权重 (默认: 1.0)")

    add_boolean_argument(
        parser,
        "ia_bce_loss",
        default=True,
        enable_help="启用 IA BCE loss",
        disable_help="禁用 IA BCE loss",
    )

    parser.add_argument("--eval-max-dets", type=int, default=500, help="评估时每张图最多保留的检测数 (默认: 500)")

    parser.add_argument("--eval-interval", type=int, default=1, help="验证间隔 epoch 数 (默认: 1)")

    add_boolean_argument(
        parser,
        "log_per_class_metrics",
        default=True,
        enable_help="记录每类别指标",
        disable_help="不记录每类别指标",
    )

    add_boolean_argument(
        parser,
        "save_val_predictions",
        default=True,
        enable_help="保存验证集预测结果",
        disable_help="不保存验证集预测结果",
    )

    add_boolean_argument(
        parser,
        "run_test",
        default=False,
        enable_help="训练完成后运行测试集评估",
        disable_help="训练完成后不运行测试集评估",
    )

    add_boolean_argument(
        parser,
        "run_eda",
        default=True,
        enable_help="训练前运行数据 EDA",
        disable_help="跳过数据 EDA",
    )

    add_boolean_argument(
        parser,
        "sync_bn",
        default=False,
        enable_help="启用 SyncBatchNorm",
        disable_help="禁用 SyncBatchNorm",
    )

    add_boolean_argument(
        parser,
        "fp16_eval",
        default=False,
        enable_help="评估时启用 FP16",
        disable_help="评估时禁用 FP16",
    )

    add_boolean_argument(
        parser,
        "dont_save_weights",
        default=False,
        enable_help="不保存训练权重",
        disable_help="保存训练权重",
    )

    add_boolean_argument(
        parser,
        "train_log_sync_dist",
        default=False,
        enable_help="分布式训练时同步日志",
        disable_help="分布式训练时不同步日志",
    )

    add_boolean_argument(
        parser,
        "train_log_on_step",
        default=False,
        enable_help="按 step 记录训练日志",
        disable_help="不按 step 记录训练日志",
    )

    add_boolean_argument(
        parser,
        "compute_val_loss",
        default=True,
        enable_help="验证时计算 loss",
        disable_help="验证时不计算 loss",
    )

    add_boolean_argument(
        parser,
        "compute_test_loss",
        default=True,
        enable_help="测试时计算 loss",
        disable_help="测试时不计算 loss",
    )

    parser.add_argument(
        "--aug-config",
        type=parse_json_argument,
        default=None,
        help='数据增强配置，传入 JSON 对象字符串，例如 \'{"HorizontalFlip": {"p": 0.5}}\'',
    )

    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="可选的类别名称列表，未提供时由数据集自动推断",
    )

    return parser.parse_args(args=list(argv) if argv is not None else None)


def get_model_from_factory(model_name: str) -> Any:
    """
    从工厂函数获取模型实例。

    Args:
        model_name: 模型名称

    Returns:
        模型实例
    """
    factory = create_model_factory()
    if model_name not in factory:
        raise ValueError(f"不支持的模型: {model_name}, 支持的模型: {list(factory.keys())}")

    return factory[model_name](layer_norm=True)


def prepare_train_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """
    准备训练参数字典。

    Args:
        args: 解析后的命令行参数

    Returns:
        Dict[str, Any]: 训练参数字典
    """
    train_kwargs = {
        "dataset_file": args.dataset_file,
        "dataset_dir": normalize_dataset_dirs(args.dataset_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "resume": args.resume,
        "eval": args.eval,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "lr_drop": args.lr_drop,
        "warmup_epochs": args.warmup_epochs,
        "lr_vit_layer_decay": args.lr_vit_layer_decay,
        "lr_component_decay": args.lr_component_decay,
        "lr_scheduler": args.lr_scheduler,
        "lr_min_factor": args.lr_min_factor,
        "clip_max_norm": args.clip_max_norm,
        "drop_path": args.drop_path,
        "output_dir": args.output_dir,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "ema_tau": args.ema_tau,
        "ema_update_interval": args.ema_update_interval,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopping_use_ema": args.early_stopping_use_ema,
        "tensorboard": args.tensorboard,
        "wandb": args.wandb,
        "mlflow": args.mlflow,
        "clearml": args.clearml,
        "project": args.project,
        "run": args.run,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "multi_scale": args.multi_scale,
        "expanded_scales": args.expanded_scales,
        "square_resize_div_64": args.square_resize_div_64,
        "do_random_resize_via_padding": args.do_random_resize_via_padding,
        "checkpoint_interval": args.checkpoint_interval,
        "group_detr": args.group_detr,
        "num_select": args.num_select,
        "ia_bce_loss": args.ia_bce_loss,
        "cls_loss_coef": args.cls_loss_coef,
        "eval_max_dets": args.eval_max_dets,
        "eval_interval": args.eval_interval,
        "log_per_class_metrics": args.log_per_class_metrics,
        "save_val_predictions": args.save_val_predictions,
        "aug_config": args.aug_config,
        "class_names": args.class_names,
        "seed": args.seed,
        "sync_bn": args.sync_bn,
        "fp16_eval": args.fp16_eval,
        "dont_save_weights": args.dont_save_weights,
        "train_log_sync_dist": args.train_log_sync_dist,
        "train_log_on_step": args.train_log_on_step,
        "compute_val_loss": args.compute_val_loss,
        "compute_test_loss": args.compute_test_loss,
        "run_test": args.run_test,
        "run_eda": args.run_eda,
        "progress_bar": args.progress_bar,
    }

    # 移除None值
    return {k: v for k, v in train_kwargs.items() if v is not None}


def normalize_dataset_dirs(raw_dataset_dirs: List[str]) -> Union[str, List[str]]:
    dataset_dirs: List[str] = []
    for raw_dir in raw_dataset_dirs:
        for part in raw_dir.split(","):
            candidate = part.strip()
            if candidate:
                dataset_dirs.append(candidate)

    if not dataset_dirs:
        raise ValueError("至少需要提供一个 --dataset-dir")

    if len(dataset_dirs) == 1:
        return dataset_dirs[0]
    return dataset_dirs


def ensure_data_yaml_in_dataset_dir(dataset_dir: str) -> None:
    dataset_path = Path(dataset_dir)
    if (dataset_path / "data.yaml").exists() or (dataset_path / "data.yml").exists():
        return

    classes_yaml = dataset_path / "classes.yaml"
    classes_yml = dataset_path / "classes.yml"

    source_file = classes_yaml if classes_yaml.exists() else classes_yml if classes_yml.exists() else None
    if source_file is None:
        return

    target_file = dataset_path / "data.yaml"
    try:
        target_file.symlink_to(source_file.name)
    except OSError:
        target_file.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")


def ensure_dataset_data_yamls(dataset_dirs: Union[str, List[str]]) -> None:
    dirs = [dataset_dirs] if isinstance(dataset_dirs, str) else dataset_dirs
    for dataset_dir in dirs:
        ensure_data_yaml_in_dataset_dir(dataset_dir)


def main():
    """主函数"""
    args = parse_arguments()
    dataset_dir_arg = normalize_dataset_dirs(args.dataset_dir)

    mode_label = "评估" if args.eval else "训练"
    print(f"开始{mode_label} RF-DETR-{args.model.capitalize()} 模型...")
    if isinstance(dataset_dir_arg, list):
        print(f"数据集目录数量: {len(dataset_dir_arg)}")
        print(f"数据集目录: {dataset_dir_arg}")
    else:
        print(f"数据集目录: {dataset_dir_arg}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 50)

    ensure_dataset_data_yamls(dataset_dir_arg)

    # 创建模型
    model = get_model_from_factory(args.model)

    # 准备训练参数
    train_kwargs = prepare_train_kwargs(args)

    # 开始训练
    model.train(**train_kwargs)

    print("训练完成!")


if __name__ == "__main__":
    main()
