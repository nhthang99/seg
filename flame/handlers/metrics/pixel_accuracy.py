from typing import Callable
import torch

import torch.nn.functional as F
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class PixelAccuracy(Metric):
    def __init__(self, mode="multi_class", output_transform: Callable=lambda x: x, smooth=1e-6):
        super(PixelAccuracy, self).__init__(output_transform)
        assert mode in ["binary", "multi_class", "multi_label"]
        self.mode = mode
        self.smooth = smooth

    def reset(self) -> None:
        self._sum = 0
        self._num_examples = 0

    def update(self, output) -> None:
        pred, target = output
        bs = pred.size(0)
        num_classes = pred.size(1)

        if self.mode == "multi_class":
            target = F.one_hot(target.long().squeeze(dim=1), num_classes)  # [B, 1, H, W] -> [B, H, W, C]
            target = target.permute(0, 3, 1, 2).contiguous() # [B, H, W, C] -> [B, C, H, W]

        pred = pred.view(bs, -1).round().long()
        target = target.view(bs, -1).long()

        pixel_acc = self.pixel_accuracy(pred, target)

        self._sum += pixel_acc
        self._num_examples += bs

    def pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor):
        corrected = torch.sum(pred & target)
        return corrected / torch.sum(target)

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError('Pixel accuracy must have at least one example before it can be computed.')
        return self._sum / self._num_examples
