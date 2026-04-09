# 2026-04-08 训练问题简记

## RFDETRBase metaclass conflict

- 现象：启动训练时报 `TypeError: metaclass conflict`，位置在 `plugin/model.py` 动态创建 `PluginRFDETRBase`。
- 原因：上游把 `RFDETRBase` 标成 deprecated 后，它变成 `deprecate.proxy._DeprecatedProxy` 包装对象；插件工厂又会提前给所有尺寸创建动态多继承类，`PluginRFDETR + DeprecatedProxy` 的 metaclass 不兼容。
- 处理：`base` 不再创建插件动态子类，直接使用上游 `RFDETRBase`；`nano/small/medium/large` 继续走插件类。

## 多数据集 dataset_dir=list 报错

- 现象：多次传 `--dataset-dir` 后报 `TypeError: expected str, bytes or os.PathLike object, not list`。
- 原因：`tools/train.py` 会把多个数据集目录规范化成 `list[str]`，但上游 `RFDETR._detect_num_classes_for_training()` / `_load_classes()` 仍按单路径处理，直接把 list 传给 `Path()`。
- 处理：让类别名加载、类别数检测和自动对齐支持 `str | list[str]`，多目录时检查类别定义一致。

## DDP 初始化阶段 CUDA OOM

- 现象：模型还没进入训练 step，就在 `nn_model.to(device)` 处 CUDA OOM。
- 原因：DDP 多进程启动时，外层 `RFDETR` 构造函数先把模型搬到默认 `cuda`，早于 Lightning 给每个 rank 分配设备，多个进程容易挤到同一张卡。
- 处理：CLI 创建 RFDETR 包装对象时固定传 `device="cpu"`；真正训练设备仍由 Lightning 的 `--device/--devices/--strategy` 接管。

## GPU0 被其他服务占用导致 OOM

- 现象：训练 step 中 `torch.OutOfMemoryError`，GPU0 只剩很少显存。
- 原因：GPU0 上已有 `VLLM::EngineCore` 进程占用约 36GB，RF-DETR rank0 再申请显存失败；其他 GPU 显存正常。
- 处理：不停止 vLLM 时用 `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7` 跑 7 卡；需要 8 卡时先释放 GPU0。

## DDP unused parameters 报错

- 现象：rank 报 `LightningModule has parameters that were not used in producing the loss`。
- 原因：RF-DETR 某些 batch / loss 路径下会有参数未参与当前 step 的 loss，普通 `ddp` 默认不允许这种情况。
- 处理：多卡默认策略改为 `ddp_find_unused_parameters_true`。

## DDP 验证指标 sync_dist warning

- 现象：验证结束打印大量 `It is recommended to use self.log(..., sync_dist=True)`。
- 原因：COCO eval callback 在 epoch 级别记录 `val/mAP_*`、`val/AP/<class>`、`val/F1` 等指标时没有告诉 Lightning 做跨卡聚合。
- 处理：在 `world_size > 1` 时给这些验证/测试指标日志加 `sync_dist=True`。

## 训练过程没有进度显示

- 现象：训练时几乎只看到验证表和 warning，看不到 batch/epoch 进度条。
- 原因：`tools/train.sh` 默认 `PROGRESS_BAR=0`，会显式传 `--no-progress-bar`。
- 处理：默认改成 `PROGRESS_BAR=1`；需要静默时再手动设置 `PROGRESS_BAR=0`。

## deprecate 兼容问题

- 现象：合并上游后出现 deprecate 相关兼容 warning/接口差异。
- 原因：上游代码使用较新的 `deprecated_class` 用法，但本地环境里的 `deprecate` 包版本和上游预期不完全一致。
- 处理：在 `variants.py` 增加兼容 shim；环境支持 `deprecated_class` 时用原生实现，不支持时退回到旧 `deprecated` 包装。
