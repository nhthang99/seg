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
        pred, target, image_infos = output  # N x 1 x H x W
        image_names, image_sizes = image_infos
        image_sizes = [(w.item(), h.item()) for w, h in zip(*image_sizes)]
        target = target.to(pred.dtype)

        for i in range(len(pred)):
            _pred, _target, image_size = pred[i:i + 1], target[i:i + 1], image_sizes[i]
            _pred = torch.nn.functional.interpolate(_pred, size=image_size[::-1], mode='bilinear', align_corners=False).round()
            if self.mode == "multi_class":
                _target = F.one_hot(_target.long().squeeze(dim=1), self.num_classes)  # [B, 1, H, W] -> [B, H, W, C]
                _target = _target.permute(0, 3, 1, 2).contiguous().to(_pred.dtype) # [B, H, W, C] -> [B, C, H, W]
            _target = torch.nn.functional.interpolate(_target, size=image_size[::-1], mode='nearest')
            n_samples = _target.shape[0]

            inter = _pred * _target

            inter = inter.reshape(inter.shape[0], -1).sum(dim=1)
            _pred = _pred.reshape(_pred.shape[0], -1).sum(dim=1)
            _target = _target.reshape(_target.shape[0], -1).sum(dim=1)

            dice = (2 * inter) / (_pred + _target + self.smooth)

            loss = 1 - dice

            self._sum += loss.item() * n_samples
            self._num_examples += n_samples

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples
