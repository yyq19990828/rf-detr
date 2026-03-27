# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""装饰器：为 __getitem__ 添加 corrupt image 重试和 debug tracing。

这两个装饰器可以叠加到任何数据集类的 __getitem__ 上，不修改上游代码。
"""

import functools
import os
import random
import sys
import time

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def with_corrupt_resilience(max_retries: int = 10):
    """装饰器：损坏图片自动重试，随机替换索引。"""

    def decorator(getitem_fn):
        @functools.wraps(getitem_fn)
        def wrapper(self, idx):
            for attempt in range(max_retries):
                try:
                    return getitem_fn(self, idx)
                except Exception:
                    logger.warning(
                        "Skipping corrupt image idx=%d, retrying with random replacement",
                        idx,
                    )
                    idx = random.randint(0, len(self) - 1)
            raise RuntimeError(f"Failed to load a valid sample after {max_retries} attempts")

        return wrapper

    return decorator


def with_debug_tracing(getitem_fn):
    """装饰器：通过 RFDETR_DEBUG_FIRST_BATCH 环境变量控制的调试追踪。

    设置 RFDETR_DEBUG_FIRST_BATCH=1 后，前 N 个样本的加载过程会打印详细日志。
    N 由 RFDETR_DEBUG_TRACE_SAMPLES 控制（默认 6）。
    """

    @functools.wraps(getitem_fn)
    def wrapper(self, idx):
        from rfdetr.utilities.distributed import get_rank

        debug_first_batch = getattr(
            self,
            "_debug_first_batch",
            os.getenv("RFDETR_DEBUG_FIRST_BATCH", "0").lower() in {"1", "true", "yes", "on"},
        )
        if not debug_first_batch:
            return getitem_fn(self, idx)

        debug_trace_samples = getattr(self, "_debug_trace_samples", int(os.getenv("RFDETR_DEBUG_TRACE_SAMPLES", "6")))
        debug_seen_samples = getattr(self, "_debug_seen_samples", 0)
        trace_this_sample = debug_seen_samples < debug_trace_samples

        if not trace_this_sample:
            return getitem_fn(self, idx)

        rank = get_rank()
        ds_name = type(self).__name__
        sample_start = time.perf_counter()
        logger.error("[RFDETR-DEBUG][rank=%d] %s __getitem__ start idx=%d", rank, ds_name, idx)
        sys.stderr.flush()

        result = getitem_fn(self, idx)

        elapsed = time.perf_counter() - sample_start
        logger.error(
            "[RFDETR-DEBUG][rank=%d] %s __getitem__ ok idx=%d elapsed=%.3fs",
            rank,
            ds_name,
            idx,
            elapsed,
        )
        sys.stderr.flush()
        self._debug_seen_samples = debug_seen_samples + 1
        return result

    return wrapper
