from typing import Callable, Optional

import torch
import torch.nn.functional as F

from flame.core.losses.base import LossBase


class DiceLoss(LossBase):
    """Dice loss from
    `Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations`
    See in detail at (https://arxiv.org/abs/1707.03237)
    Args:
        LossBase (_type_): _description_
    """
    def __init__(self, mode="binary", num_classes: Optional[int]=None,
                 output_transform: Callable=lambda x: x) -> None:
        super(DiceLoss, self).__init__(output_transform)
        assert mode in ["binary", "multi_class", "multi_label"]
        self.mode = mode
        if mode == "multi_class" and isinstance(num_classes, int):
            raise AttributeError("Attribute `num_classes` should be integer during multi-class mode"
                                 f"instead of {type(num_classes)}")
        self.num_classes = num_classes # Only used in multi-class mode

    def init(self):
        return super().init()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss implementation
        Supported modes: binary, multi-class and multi-label
        Args:
            pred (torch.Tensor): Prediction mask [B, C, H, W]
            target (torch.Tensor): Target mask [B, C, H, W] or [B, H, W]
        """
        assert pred.size() == target.size()
        pred = pred.to(target.dtype)

        if self.mode == "multi_class" and target.dim() == 3:
            target = F.one_hot(target, self.num_classes) # [B, H, W] -> [B, H, W, C]
            target = target.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]

        bs = pred.size(0)
        num_classes = pred.size(1)
        pred = pred.view(bs, num_classes, -1)
        target = target.view(bs, num_classes, -1)

        dice_score = self.dice_score(pred, target)
        loss = self.aggregate_loss(1 - dice_score)

        return loss

    def aggregate_loss(self, loss: torch.Tensor):
        return loss.mean()

    def dice_score(self, pred: torch.Tensor, target: torch.Tensor, smooth: float=1e-7):
        assert pred.size() == target.size()
        inter = pred * target
        dice_score = (2 * inter + smooth) / (pred + target + smooth)
        return dice_score
