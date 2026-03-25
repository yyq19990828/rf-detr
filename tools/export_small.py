# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# This file contains code licensed under the Apache License, Version 2.0.
# See NOTICE for more details.

import argparse

import onnx
import onnxsim
import torch

from rfdetr import RFDETRSmall


def export_model(args):

    # Determine the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = RFDETRSmall(device="cpu", pretrain_weights=args.checkpoint)
    my_model = model.model.model
    my_model.to(device)

    # Create a dummy input tensor for exporting the model
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    # Run a forward pass to check output structure
    output = my_model(dummy_input)
    print("Model output keys:", output.keys())

    # Move the model to export mode
    my_model.export()

    # Export the model
    print(f"Exporting model to {args.model_name}...")
    torch.onnx.export(
        my_model,
        dummy_input,
        args.model_name,
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        input_names=["images"],
        output_names=["pred_boxes", "pred_logits"],
        dynamo=False,
    )

    # Fix IR version to 10
    onnx_model = onnx.load(args.model_name)
    onnx_model.ir_version = 10
    onnx.save(onnx_model, args.model_name)
    print(f"Model exported as {args.model_name} (IR version: 10)")

    # Simplify ONNX model with onnxsim
    print("Simplifying ONNX model with onnxsim...")
    input_data = {"images": dummy_input.detach().cpu().numpy()}
    simplified_model, check_ok = onnxsim.simplify(
        onnx_model,
        check_n=0,
        input_data=input_data,
        dynamic_input_shape=False,
    )
    sim_path = args.model_name.replace(".onnx", "_sim.onnx")
    simplified_model.ir_version = 10
    onnx.save(simplified_model, sim_path)
    print(f"Simplified model saved as {sim_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RFDETRSmall model to ONNX format.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="rf-detr-small.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="rfdetr-small.onnx",
        help="Name of the output ONNX model file",
    )
    args = parser.parse_args()

    export_model(args)
