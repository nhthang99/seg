from typing import Callable

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
    def __init__(self, output_transform: Callable=lambda x: x) -> None:
        super(BCELoss, self).__init__(output_transform)

    def init(self):
        return super().init()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary cross entropy loss

        Args:
            pred (torch.Tensor): Prediction mask [B, C, H, W] or [B, H, W]
            target (torch.Tensor): Target mask [B, C, H, W] or [B, H, W]

        Returns:
            float: Binary cross entropy loss
        """
        loss = F.binary_cross_entropy(pred, target, reduction="mean")
        return loss
