# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

import rfdetr.main as main_module
from rfdetr.main import Model
from rfdetr.util.misc import NestedTensor


class _TinyTrainDataset(Dataset):
    """Minimal dataset that yields a single train/val sample."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int):
        samples = torch.ones((3, 4, 4), dtype=torch.float32)
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "orig_size": torch.tensor([4, 4], dtype=torch.int64),
            "size": torch.tensor([4, 4], dtype=torch.int64),
            "image_id": torch.tensor(1, dtype=torch.int64),
        }
        return samples, target


class _TinyTrainModel(nn.Module):
    """Small trainable module used to keep the legacy train loop lightweight."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, samples: NestedTensor, _targets):
        return {"pred": self.weight * samples.tensors.mean()}

    def update_drop_path(self, _value: float, _layers: int) -> None:
        return None

    def update_dropout(self, _value: float) -> None:
        return None


class _TinyCriterion(nn.Module):
    """Criterion stub with the keys expected by the training loop."""

    def __init__(self) -> None:
        super().__init__()
        self.weight_dict = {"loss_bbox": 1.0, "class_error": 1.0}

    def forward(self, outputs, _targets):
        base = outputs["pred"]
        return {"loss_bbox": base * 0 + 1.0, "class_error": base * 0 + 0.0}


class _FakeDDP(nn.Module):
    """Tiny DistributedDataParallel stand-in for unit tests."""

    def __init__(self, module: nn.Module, **_kwargs) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _FakeDistributedSampler(Sampler[int]):
    """Sampler stub that mimics the minimal DistributedSampler API."""

    def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
        self._dataset = dataset
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self._dataset)))

    def __len__(self) -> int:
        return len(self._dataset)

    def set_epoch(self, _epoch: int) -> None:
        return None


class _StopTraining(RuntimeError):
    """Sentinel exception used to stop the train loop once synchronization is observed."""


def test_distributed_train_waits_for_eda_before_epoch_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Distributed training should synchronize after rank-0 EDA before entering the epoch loop."""
    events: list[str] = []
    tiny_dataset = _TinyTrainDataset()

    def fake_init_distributed_mode(args) -> None:
        args.distributed = True
        args.rank = 0
        args.gpu = 0
        args.world_size = 2
        args.dist_backend = "nccl"

    def fake_build_dataset(*_args, **_kwargs) -> _TinyTrainDataset:
        return tiny_dataset

    def fake_run_eda(*_args, **_kwargs) -> None:
        events.append("run_eda")

    def fake_barrier() -> None:
        events.append("barrier")

    def fake_train_one_epoch(*_args, **_kwargs):
        events.append("train")
        raise _StopTraining

    monkeypatch.setattr(main_module, "build_model", lambda _args: _TinyTrainModel())
    monkeypatch.setattr(main_module, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(main_module, "build_criterion_and_postprocessors", lambda _args: (_TinyCriterion(), {}))
    monkeypatch.setattr(main_module, "get_coco_api_from_dataset", lambda _dataset: object())
    monkeypatch.setattr(
        main_module,
        "get_param_dict",
        lambda _args, model_without_ddp: [{"params": model_without_ddp.weight, "lr": 1e-4}],
    )
    monkeypatch.setattr(main_module, "benchmark", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(main_module, "DistributedSampler", _FakeDistributedSampler)
    monkeypatch.setattr(main_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(main_module, "evaluate", lambda *_args, **_kwargs: ({}, object()))
    monkeypatch.setattr(main_module, "is_main_process", lambda: True)
    monkeypatch.setattr(main_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(main_module.utils, "get_sha", lambda: "sha: test")
    monkeypatch.setattr(main_module.utils, "init_distributed_mode", fake_init_distributed_mode)
    monkeypatch.setattr(main_module.torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    monkeypatch.setattr(main_module.torch.distributed, "barrier", fake_barrier)
    monkeypatch.setattr("rfdetr.datasets.eda.run_eda", fake_run_eda)

    model = Model(device="cpu", pretrain_weights=None, num_classes=1, segmentation_head=False)

    with pytest.raises(_StopTraining):
        model.train(
            callbacks=defaultdict(list),
            dataset_dir=str(tmp_path),
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            num_workers=0,
            class_names=["object"],
            device="cpu",
            output_dir=str(tmp_path / "output"),
            run_eda=True,
            run_test=False,
            progress_bar=False,
            use_ema=False,
        )

    assert events == ["run_eda", "barrier", "train"]
