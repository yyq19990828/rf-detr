# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# This file contains code licensed under the Apache License, Version 2.0.
# See NOTICE for more details.

import argparse

import torch

from rfdetr import RFDETRMedium


def export_model(model_name: str):

    # Determine the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = RFDETRMedium(
        device="cpu",
        pretrain_weights="/home/hehao/桌面/rf-detr/output/medium_上电室内_20260226/checkpoint_best_ema.pth",
    )
    my_model = model.model.model
    my_model.to(device)

    # Create a dummy input tensor for exporting the model
    dummy_input = torch.randn(1, 3, 576, 576).to(device)  # Example input: batch size 4, 3 channels, 576x576

    # Run a forward pass to check output structure
    output = my_model(dummy_input)
    print("Model output keys:", output.keys())

    # Move the model to export mode
    my_model.export()

    # Export the model
    print(f"Exporting model to {model_name}...")
    torch.onnx.export(
        my_model,  # Model to export
        dummy_input,  # Dummy input tensor
        model_name,  # Output ONNX file name
        export_params=True,  # Store trained parameters
        opset_version=17,  # ONNX version
        do_constant_folding=False,  # Optimize constant expressions
        input_names=["images"],  # Define input tensor name
        output_names=["pred_boxes", "pred_logits"],  # Use first two keys as output names
        dynamo=False,  # Use legacy exporter for PyTorch 2.9 compatibility
    )

    print(f"Model successfully exported as {model_name}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Export RFDETRBase model to ONNX format.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="rfdetr-medium_上电室内_20260226.onnx",
        help="Name of the output ONNX model file",
    )
    args = parser.parse_args()

    export_model(args.model_name)
