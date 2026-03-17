# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the standalone tools/train.py CLI helper."""

from tools.train import parse_arguments, prepare_train_kwargs


class TestToolsTrainCli:
    """tools/train.py should stay aligned with TrainConfig-facing options."""

    def test_parse_arguments_uses_current_train_defaults(self):
        """CLI defaults should match the current TrainConfig defaults for key flags."""
        args = parse_arguments(["--dataset-dir", "datasets/demo"])

        assert args.dataset_file == "roboflow"
        assert args.use_ema is False
        assert args.multi_scale is True
        assert args.expanded_scales is True
        assert args.square_resize_div_64 is True
        assert args.progress_bar is False
        assert args.run_test is False
        assert args.run_eda is True
        assert args.eval is False
        assert args.tensorboard is True
        assert args.early_stopping is False

    def test_prepare_train_kwargs_forwards_latest_train_options(self):
        """prepare_train_kwargs should forward newly added TrainConfig fields."""
        args = parse_arguments(
            [
                "--dataset-dir",
                "datasets/a",
                "--dataset-dir",
                "datasets/b",
                "--dataset-file",
                "yolo",
                "--resume",
                "output/checkpoint.pth",
                "--eval",
                "--seed",
                "123",
                "--use-ema",
                "--ema-tau",
                "250",
                "--ema-update-interval",
                "3",
                "--early-stopping",
                "--early-stopping-use-ema",
                "--mlflow",
                "--clearml",
                "--prefetch-factor",
                "4",
                "--pin-memory",
                "--persistent-workers",
                "--no-expanded-scales",
                "--no-square-resize-div-64",
                "--do-random-resize-via-padding",
                "--group-detr",
                "7",
                "--num-select",
                "128",
                "--cls-loss-coef",
                "2.5",
                "--no-ia-bce-loss",
                "--eval-max-dets",
                "300",
                "--eval-interval",
                "2",
                "--no-log-per-class-metrics",
                "--no-save-val-predictions",
                "--aug-config",
                '{"HorizontalFlip": {"p": 0.5}}',
                "--class-names",
                "car",
                "bus",
                "truck",
                "--sync-bn",
                "--fp16-eval",
                "--dont-save-weights",
                "--train-log-sync-dist",
                "--train-log-on-step",
                "--no-compute-val-loss",
                "--no-compute-test-loss",
                "--run-test",
                "--no-run-eda",
                "--progress-bar",
            ]
        )

        train_kwargs = prepare_train_kwargs(args)

        assert train_kwargs["dataset_dir"] == ["datasets/a", "datasets/b"]
        assert train_kwargs["dataset_file"] == "yolo"
        assert train_kwargs["resume"] == "output/checkpoint.pth"
        assert train_kwargs["eval"] is True
        assert train_kwargs["seed"] == 123
        assert train_kwargs["use_ema"] is True
        assert train_kwargs["ema_tau"] == 250
        assert train_kwargs["ema_update_interval"] == 3
        assert train_kwargs["early_stopping"] is True
        assert train_kwargs["early_stopping_use_ema"] is True
        assert train_kwargs["mlflow"] is True
        assert train_kwargs["clearml"] is True
        assert train_kwargs["prefetch_factor"] == 4
        assert train_kwargs["pin_memory"] is True
        assert train_kwargs["persistent_workers"] is True
        assert train_kwargs["expanded_scales"] is False
        assert train_kwargs["square_resize_div_64"] is False
        assert train_kwargs["do_random_resize_via_padding"] is True
        assert train_kwargs["group_detr"] == 7
        assert train_kwargs["num_select"] == 128
        assert train_kwargs["cls_loss_coef"] == 2.5
        assert train_kwargs["ia_bce_loss"] is False
        assert train_kwargs["eval_max_dets"] == 300
        assert train_kwargs["eval_interval"] == 2
        assert train_kwargs["log_per_class_metrics"] is False
        assert train_kwargs["save_val_predictions"] is False
        assert train_kwargs["aug_config"] == {"HorizontalFlip": {"p": 0.5}}
        assert train_kwargs["class_names"] == ["car", "bus", "truck"]
        assert train_kwargs["sync_bn"] is True
        assert train_kwargs["fp16_eval"] is True
        assert train_kwargs["dont_save_weights"] is True
        assert train_kwargs["train_log_sync_dist"] is True
        assert train_kwargs["train_log_on_step"] is True
        assert train_kwargs["compute_val_loss"] is False
        assert train_kwargs["compute_test_loss"] is False
        assert train_kwargs["run_test"] is True
        assert train_kwargs["run_eda"] is False
        assert train_kwargs["progress_bar"] is True
