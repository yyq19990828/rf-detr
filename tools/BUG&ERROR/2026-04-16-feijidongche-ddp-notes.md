# 2026-04-16 非机动车车牌多卡训练问题记录

## DDP 验证阶段 CPU metric all_reduce 崩溃

- 现象：双卡训练在 `Sanity Checking` 结束后的验证汇总阶段报错：
  `RuntimeError: No backend type associated with device type cpu`。
- 典型栈位置：
  `src/rfdetr/training/callbacks/coco_eval.py`
  `src/rfdetr/training/callbacks/best_model.py`
  `pytorch_lightning/.../result.py`
  `torch.distributed.all_reduce(...)`
- 原因：`COCOEvalCallback` 和 `RFDETREarlyStopping` 会把 epoch 级指标直接写入
  `trainer.callback_metrics`，其中一部分是 CPU tensor：
  `metrics[...].detach().cpu()` / `torch.tensor(effective)`。
  在 DDP 下，Lightning 会尝试对这些指标做跨卡归约，最终对 CPU tensor 调用
  `all_reduce`，于是报错。
- 处理：
  - 在 `src/rfdetr/training/callbacks/coco_eval.py` 增加 `_to_metric_device()`，
    把 mAP / mAR / F1 / precision / recall / segm 指标统一转到 `pl_module.device`。
  - 在 `src/rfdetr/training/callbacks/best_model.py` 中把 synthetic monitor
    `__rfdetr_effective_map__` 也改成创建在 `pl_module.device` 上。

## DDP 验证阶段 per-class AP 仍触发同类崩溃

- 现象：修完总指标后，训练能走完 sanity-check 表格打印，但仍在
  `logger_connector.on_epoch_end()` 阶段报同类 DDP reduce 错误。
- 原因：`COCOEvalCallback._build_per_class_rows()` 中还有一处
  `pl_module.log(f"{split}/AP/{name}", ap, sync_dist=sync_dist)`，
  这里的 `ap` 仍是原始 tensor，未做设备对齐。
- 处理：把 per-class AP 日志也改成
  `self._to_metric_device(ap, pl_module)` 后再 `pl_module.log(...)`。

## 多卡训练 epoch 级 train/* 指标 sync_dist 告警

- 现象：训练启动后出现 Lightning 告警：
  - `It is recommended to use self.log('train/cardinality_error_enc', ..., sync_dist=True)`
  - `It is recommended to use self.log('train/loss', ..., sync_dist=True)`
- 原因：`src/rfdetr/training/module_model.py` 中训练损失日志本来就支持
  `sync_dist=train_log_sync_dist`，但 `TrainConfig.train_log_sync_dist`
  默认值是 `False`，多卡训练时 epoch 级 `train/*` 指标不会自动跨卡聚合。
- 处理：
  - 在 `tools/train_feijidongche.sh` 默认加入 `TRAIN_LOG_SYNC_DIST=1`
  - 在 `tools/train.sh` 同步加入 `TRAIN_LOG_SYNC_DIST=1`
  - shell 脚本会显式传入 `--train-log-sync-dist`

## 验证阶段 `RuntimeWarning: invalid value encountered in divide`

- 现象：sanity-check 验证阶段偶发 numpy warning：
  `RuntimeWarning: invalid value encountered in divide`
- 判断：当前更像评估链路中的空分母/空数组 warning，而不是训练数值爆炸。
  当时验证表所有指标都是 `0.0000`，说明样本极少、预测为空或没有有效 TP，
  在 COCO/torchmetrics 评估链路里很容易触发 `0/0` 或空统计。
- 当前处理：暂未修改训练逻辑；若后续每个 epoch 都持续出现，再单独排查
  验证集标注、类别映射、YOLO 标签质量与空框问题。

## 为什么以前没碰到这个 bug

- 以前大概率主要跑的是单卡。单卡不会触发 DDP 的指标归约，因此 CPU tensor
  写进 `callback_metrics` / `logged_metrics` 也不会立刻炸。
- 这次把 `tools/train_feijidongche.sh` 完整对齐到 `tools/train.sh` 后，真正启用了：
  - `--devices 2`
  - `--strategy ddp_find_unused_parameters_true`
  多卡验证 sanity-check 才把这条旧隐患暴露出来。
- 此类问题只会在“多卡 + epoch 级指标日志/回调指标 + Lightning 归约”组合下出现，
  不是普通前向或反向训练本身的问题。
