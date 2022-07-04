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
    def __init__(self, mode, num_classes: Optional[int]=None,
                 output_transform: Callable=lambda x: x) -> None:
        super(DiceLoss, self).__init__(mode=mode, num_classes=num_classes, output_transform=output_transform)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss implementation
        Supported modes: binary, multi-class and multi-label
        Args:
            pred (torch.Tensor): Prediction mask [B, C, H, W]
            target (torch.Tensor): Target mask [B, C, H, W] or [B, H, W]
        """
        if self.mode == "multi_class":
            target = F.one_hot(target.long().squeeze(dim=1), self.num_classes)  # [B, 1, H, W] -> [B, H, W, C]
            target = target.permute(0, 3, 1, 2).contiguous() # [B, H, W, C] -> [B, C, H, W]

        bs = pred.size(0)
        pred = pred.view(bs, -1)
        target = target.view(bs, -1)

        dice_score = self.dice_score(pred, target)
        loss = self.aggregate_loss(1 - dice_score)

        return loss

    def aggregate_loss(self, loss: torch.Tensor):
        return loss.mean()

    def dice_score(self, pred: torch.Tensor, target: torch.Tensor, smooth: float=1e-7):
        assert pred.size() == target.size()
        inter = torch.sum(pred * target)
        dice_score = (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)
        return dice_score
