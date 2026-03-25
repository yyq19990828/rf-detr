"""增强版 CocoDetection：添加 corrupt resilience + debug tracing。"""

from rfdetr.datasets.coco import CocoDetection

from plugin.datasets.resilience import with_corrupt_resilience, with_debug_tracing


class RobustCocoDetection(CocoDetection):
    """在上游 CocoDetection 基础上叠加 corrupt image 重试和 debug tracing。"""

    @with_corrupt_resilience(max_retries=10)
    @with_debug_tracing
    def __getitem__(self, idx):
        return super().__getitem__(idx)
