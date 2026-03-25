"""
RF-DETR 非侵入式自定义扩展插件。

所有自定义增强功能集中在此目录中，上游代码保持零修改。
上游更新时只需在此目录中适配，不会产生 merge 冲突。

功能清单:
- corrupt image resilience: 数据加载时自动跳过损坏图片并重试
- debug tracing: RFDETR_DEBUG_FIRST_BATCH 环境变量控制的调试追踪
- YOLO annotation caching: 大规模数据集标注缓存加速
- EDA: 训练前自动运行探索性数据分析
- multi-dir class validation: 多目录训练时类名一致性校验
- val_visualizer: 验证集预测可视化 callback
- test split fallback: test/ 不存在时回退到 val/
"""
