import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class Dice(Metric):
    def __init__(self, smooth=1e-6, output_transform=lambda x: x):
        super(Dice, self).__init__(output_transform)
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
