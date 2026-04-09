#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unified RF-DETR ONNX export helper for all detection model sizes."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import onnx
import onnxsim
import torch
from onnx import ModelProto

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


MODEL_ALIASES = {
    "base": "base",
    "rfdetrbase": "base",
    "pluginrfdetrbase": "base",
    "rfdetrbaseconfig": "base",
    "nano": "nano",
    "rfdetrnano": "nano",
    "pluginrfdetrnano": "nano",
    "rfdetrnanoconfig": "nano",
    "small": "small",
    "rfdetrsmall": "small",
    "pluginrfdetrsmall": "small",
    "rfdetrsmallconfig": "small",
    "medium": "medium",
    "rfdetrmedium": "medium",
    "pluginrfdetrmedium": "medium",
    "rfdetrmediumconfig": "medium",
    "large": "large",
    "rfdetrlarge": "large",
    "pluginrfdetrlarge": "large",
    "rfdetrlargeconfig": "large",
}


def normalize_model_name(value: str | None) -> str | None:
    """Normalize model metadata or CLI aliases to a supported model size."""
    if value is None:
        return None
    key = Path(value).stem.replace("-", "").replace("_", "").strip().lower()
    return MODEL_ALIASES.get(key)


def checkpoint_args_get(args: Any, key: str) -> Any:
    """Read a value from checkpoint args stored as a dict or argparse namespace."""
    if isinstance(args, Mapping):
        return args.get(key)
    return getattr(args, key, None)


def find_training_config(checkpoint_path: Path, explicit_path: str | None) -> Path | None:
    """Return the training config path associated with a checkpoint, if present."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"training_config not found: {path}")
        return path

    candidate = checkpoint_path.parent / "training_config.json"
    return candidate if candidate.exists() else None


def load_training_config(path: Path | None) -> dict[str, Any]:
    """Load training_config.json metadata when it is available."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint_metadata(checkpoint_path: str, training_config_path: str | None = None) -> dict[str, Any]:
    """Collect model metadata from .pth/.ckpt payloads and optional training_config.json."""
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict payload, got {type(checkpoint).__name__}: {path}")
    training_config = load_training_config(find_training_config(path, training_config_path))

    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model_config = training_config.get("model_config", {})

    model_name = (
        checkpoint.get("model_name")
        or checkpoint_args_get(args, "model_name")
        or model_config.get("model_name")
        or training_config.get("model_config_type")
    )
    pretrain_weights = checkpoint_args_get(args, "pretrain_weights") or model_config.get("pretrain_weights")

    class_names = checkpoint_args_get(args, "class_names") or training_config.get("class_names")
    num_classes = (
        checkpoint_args_get(args, "num_classes")
        or training_config.get("num_classes")
        or model_config.get("num_classes")
        or (len(class_names) if isinstance(class_names, list) else None)
    )
    resolution = checkpoint_args_get(args, "resolution") or model_config.get("resolution")

    if normalize_model_name(str(model_name) if model_name is not None else None) is None and pretrain_weights:
        model_name = Path(str(pretrain_weights)).name

    return {
        "checkpoint": checkpoint,
        "model": normalize_model_name(str(model_name) if model_name is not None else None),
        "num_classes": num_classes,
        "resolution": resolution,
    }


def get_model_class(model_name: str) -> type:
    """Return the RF-DETR class for a supported detection model size."""
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    model_classes = {
        "base": RFDETRBase,
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    return model_classes[model_name]


def extract_model_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Extract an RF-DETR module state dict from .pth or Lightning .ckpt payloads."""
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        state_dict = checkpoint["state_dict"]
        prefix = "model."
        return {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix) and isinstance(value, torch.Tensor)
        }
    raise KeyError("Checkpoint must contain either a 'model' or 'state_dict' mapping.")


def build_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str | None,
    num_classes: int | None,
    training_config_path: str | None,
) -> tuple[Any, str, int]:
    """Instantiate a model from CLI options plus checkpoint metadata."""
    metadata = load_checkpoint_metadata(checkpoint_path, training_config_path)
    resolved_model = normalize_model_name(model_name) or metadata["model"]
    if resolved_model is None:
        raise ValueError("无法从 checkpoint metadata 推断模型尺寸，请显式传入 --model nano|small|medium|large|base")

    resolved_num_classes = num_classes or metadata["num_classes"]
    constructor_kwargs: dict[str, Any] = {"device": "cpu", "pretrain_weights": None}
    if resolved_num_classes is not None:
        constructor_kwargs["num_classes"] = int(resolved_num_classes)

    model = get_model_class(resolved_model)(**constructor_kwargs)
    state_dict = extract_model_state_dict(metadata["checkpoint"])
    missing, unexpected = model.model.model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)} keys")

    resolution = metadata["resolution"] or getattr(model.model, "resolution", None) or model.model_config.resolution
    return model, resolved_model, int(resolution)


def set_dynamic_batch_dim(onnx_model: ModelProto, dim_param: str = "batch") -> ModelProto:
    """Mark the first dimension of model inputs and outputs as dynamic batch."""
    for value_info in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        tensor_shape = value_info.type.tensor_type.shape
        if tensor_shape.dim:
            tensor_shape.dim[0].dim_value = 0
            tensor_shape.dim[0].dim_param = dim_param
    return onnx_model


def simplify_onnx_model(model_path: str, input_tensor: torch.Tensor, dynamic_batch: bool) -> str:
    """Simplify an ONNX model with onnxsim and preserve IR v10."""
    onnx_model = onnx.load(model_path)
    input_data = {"images": input_tensor.detach().cpu().numpy()}
    simplified_model, check_ok = onnxsim.simplify(
        onnx_model,
        check_n=0,
        input_data=input_data,
        dynamic_input_shape=dynamic_batch,
    )
    if not check_ok:
        raise RuntimeError("Failed to simplify ONNX model.")
    if dynamic_batch:
        simplified_model = set_dynamic_batch_dim(simplified_model)
    simplified_model.ir_version = 10
    sim_path = model_path.replace(".onnx", "_sim.onnx")
    onnx.save(simplified_model, sim_path)
    return sim_path


def export_model(args: argparse.Namespace) -> str:
    """Export an RF-DETR checkpoint to ONNX."""
    model, resolved_model, resolution = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        num_classes=args.num_classes,
        training_config_path=args.training_config,
    )
    if args.resolution is not None:
        resolution = args.resolution

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    my_model = model.model.model.to(device)
    my_model.eval()

    dummy_input = torch.randn(args.batch_size, 3, resolution, resolution, device=device)
    output = my_model(dummy_input)
    print("Model output keys:", output.keys())
    my_model.export()

    output_path = args.output or f"rfdetr-{resolved_model}.onnx"
    Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "pred_logits": {0: "batch"},
        }

    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    print(f"Exporting {resolved_model} ({resolution}x{resolution}) to {output_path}...")
    torch.onnx.export(
        my_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=False,
        input_names=["images"],
        output_names=["pred_boxes", "pred_logits"],
        dynamic_axes=dynamic_axes,
        **export_kwargs,
    )

    onnx_model = onnx.load(output_path)
    if args.dynamic_batch:
        onnx_model = set_dynamic_batch_dim(onnx_model)
    onnx_model.ir_version = args.ir_version
    onnx.save(onnx_model, output_path)
    print(f"Model exported as {output_path} (IR version: {args.ir_version})")

    if args.simplify:
        try:
            sim_path = simplify_onnx_model(output_path, dummy_input, args.dynamic_batch)
            print(f"Simplified model saved as {sim_path}")
        except RuntimeError as exc:
            if args.dynamic_batch:
                print(f"Skipping ONNX simplification for dynamic batch export: {exc}")
            else:
                raise

    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the unified export script."""
    parser = argparse.ArgumentParser(description="Export any RF-DETR detection model size to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth or Lightning .ckpt checkpoint")
    parser.add_argument(
        "--model",
        choices=["base", "nano", "small", "medium", "large"],
        help="Model size; inferred from checkpoint metadata when omitted",
    )
    parser.add_argument("--num-classes", type=int, help="Class count; inferred from metadata when omitted")
    parser.add_argument("--resolution", type=int, help="Square export resolution; inferred from metadata when omitted")
    parser.add_argument("--training-config", help="Path to training_config.json for .ckpt metadata")
    parser.add_argument("--output", "--model_name", dest="output", help="Output ONNX path")
    parser.add_argument("--device", help="Export device, defaults to cuda when available")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--ir-version", type=int, default=10, help="ONNX IR version to write")
    parser.add_argument("--dynamic-batch", action="store_true", help="Mark batch dimension as dynamic")
    parser.add_argument("--simplify", action="store_true", help="Run onnxsim after export")
    return parser.parse_args(argv)


if __name__ == "__main__":
    export_model(parse_args())
