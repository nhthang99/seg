from typing import Callable, Optional

import torch
import torch.nn.functional as F

from flame.core.losses.base import LossBase


class BCELoss(LossBase):
    """Binary cross entropy loss from
    `Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels`
    See in detail at (https://arxiv.org/abs/1805.07836)

    Args:
        LossBase (_type_): _description_
    """
    def __init__(self, mode, num_classes: Optional[int]=None,
                 output_transform: Callable=lambda x: x) -> None:
        super(BCELoss, self).__init__(mode=mode, num_classes=num_classes, output_transform=output_transform)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary cross entropy loss

        Args:
            pred (torch.Tensor): Prediction mask [B, C, H, W] or [B, H, W]
            target (torch.Tensor): Target mask [B, C, H, W] or [B, H, W]

        Returns:
            float: Binary cross entropy loss
        """
        if self.mode == "multi_class":
            target = F.one_hot(target.long().squeeze(dim=1), self.num_classes)  # [B, 1, H, W] -> [B, H, W, C]
            target = target.permute(0, 3, 1, 2).contiguous().to(pred.dtype) # [B, H, W, C] -> [B, C, H, W]
        loss = F.binary_cross_entropy(pred, target, reduction="mean")
        return loss
