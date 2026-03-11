# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# This file contains code licensed under the Apache License, Version 2.0.
# See NOTICE for more details.

import argparse
import os

import onnx
import onnxsim
import torch

from rfdetr import RFDETRBase
from rfdetr.deploy._onnx import OnnxOptimizer


def export_model(args):

    # Determine the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = RFDETRBase(device="cpu", pretrain_weights=args.checkpoint)
    my_model = model.model.model
    my_model.to(device)

    # Create a dummy input tensor for exporting the model
    dummy_input = torch.randn(1, 3, 560, 560).to(device)  # Example input: batch size 1, 3 channels, 560x560

    # Run a forward pass to check output structure
    output = my_model(dummy_input)
    print("Model output keys:", output.keys())

    # Move the model to export mode
    my_model.export()

    # Export the model using legacy ONNX export (not torch.export)
    print("Trying fixed batch size...")
    # Try fixed batch size as fallback
    torch.onnx.export(
        my_model,
        dummy_input,
        args.model_name,
        input_names=["images"],
        output_names=["pred_boxes", "pred_logits"],
        opset_version=17,
    )
    print(f"Model successfully exported as {args.model_name} (fixed batch size)")

    # Simplify ONNX model if requested
    if args.simplify:
        print("Simplifying ONNX model...")
        simplified_path = onnx_simplify(args.model_name, ["images"], dummy_input, force=args.force)
        print(f"Simplified model saved as {simplified_path}")


def onnx_simplify(onnx_dir: str, input_names, input_tensors, force=False):
    """
    Simplify ONNX model using onnxsim and OnnxOptimizer
    """
    sim_onnx_dir = onnx_dir.replace(".onnx", "_sim.onnx")
    if os.path.isfile(sim_onnx_dir) and not force:
        return sim_onnx_dir

    if isinstance(input_tensors, torch.Tensor):
        input_tensors = [input_tensors]

    print(f"Start simplifying ONNX model: {onnx_dir}")
    opt = OnnxOptimizer(onnx_dir)
    opt.info("Model: original")
    opt.insert_convx_layernorm_plugin()
    opt.common_opt()
    opt.info("Model: optimized")
    opt.save_onnx(sim_onnx_dir)

    # Create input dictionary for onnxsim
    input_dict = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, input_tensors)}

    # Apply onnxsim
    model_opt, check_ok = onnxsim.simplify(onnx_dir, check_n=3, input_data=input_dict, dynamic_input_shape=False)

    if check_ok:
        onnx.save(model_opt, sim_onnx_dir)
        print(f"Successfully simplified ONNX model: {sim_onnx_dir}")
    else:
        raise RuntimeError("Failed to simplify ONNX model.")

    return sim_onnx_dir


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Export RFDETRMedium model to ONNX format.")
    parser.add_argument(
        "--checkpoint", type=str, default="0829_base/checkpoint_best_ema.pth", help="Path to the model checkpoint"
    )
    parser.add_argument("--model_name", type=str, default="best_ema.onnx", help="Name of the output ONNX model file")
    parser.add_argument("--simplify", action="store_true", help="Simplify the exported ONNX model")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing simplified model")
    args = parser.parse_args()

    export_model(args)
