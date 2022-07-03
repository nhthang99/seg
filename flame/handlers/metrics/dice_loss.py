from typing import Optional, Callable

import torch
import torch.nn.functional as F

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class Dice(Metric):
    def __init__(self, mode, num_classes: Optional[int]=None,
                 output_transform: Callable=lambda x: x, smooth=1e-6) -> None:
        super(Dice, self).__init__(output_transform=output_transform)
        assert mode in ["binary", "multi_class", "multi_label"]
        self.mode = mode
        if mode == "multi_class" and not isinstance(num_classes, int):
            raise AttributeError("Attribute `num_classes` should be integer during multi-class mode"
                                 f"instead of {type(num_classes)}")
        self.num_classes = num_classes # Only used in multi-class mode
        self.smooth = smooth

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        """Dice loss calculation for each iteration
        Supported modes: binary, multi-class and multi-label
        Args:
            pred (torch.Tensor): Prediction mask [B, C, H, W]
            target (torch.Tensor): Target mask [B, C, H, W] or [B, H, W]
        """
        pred, target = output
        if self.mode == "multi_class":
            target = F.one_hot(target.long().squeeze(dim=1), self.num_classes)  # [B, 1, H, W] -> [B, H, W, C]
            target = target.permute(0, 3, 1, 2).contiguous() # [B, H, W, C] -> [B, C, H, W]
        target = target.to(pred.dtype)

        bs = pred.size(0)
        pred = pred.view(bs, -1)
        target = target.view(bs, -1)

        dice_score = self.dice_score(pred, target)
        loss = 1 - dice_score

        self._sum += torch.sum(loss)
        self._num_examples += bs

    def dice_score(self, pred: torch.Tensor, target: torch.Tensor, smooth: float=1e-7):
        assert pred.size() == target.size()
        inter = torch.sum(pred * target)
        dice_score = (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)
        return dice_score

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')
        if self._num_examples == 0:
            return 0
        return self._sum / self._num_examples
