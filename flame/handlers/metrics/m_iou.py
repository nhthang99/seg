from typing import Callable, Optional
import torch

import torch.nn.functional as F
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class MeanIOU(Metric):
    def __init__(self, mode="multi_class", num_classes: Optional[int]=None,
                 output_transform: Callable=lambda x: x, smooth=1e-6):
        super(MeanIOU, self).__init__(output_transform)
        assert mode in ["binary", "multi_class", "multi_label"]
        self.mode = mode
        self.smooth = smooth
        self.num_classes = num_classes

    def reset(self) -> None:
        pass

    def started(self) -> None:
        self._ious = torch.zeros(0, self.num_classes)

    def update(self, output) -> None:
        pred, target = output
        num_classes = pred.size(1)

        if self.mode == "multi_class":
            target = F.one_hot(target.long().squeeze(dim=1), num_classes)  # [B, 1, H, W] -> [B, H, W, C]
            target = target.permute(0, 3, 1, 2).contiguous() # [B, H, W, C] -> [B, C, H, W]

        mious = []
        for pred_, target_ in zip(pred, target):
            miou = []
            for i in range(self.num_classes):
                p = pred_[i, ...].round().long()
                t = target_[i, ...].long()
                inter = (p * t).sum()
                union = (p.sum() + t.sum() - inter + self.smooth)
                miou.append(inter / union)
            mious.append(miou)
        self._ious = torch.cat(self._ious, torch.tensor(mious))

    def compute(self) -> float:
        return torch.mean(self._ious).item()
