from typing import Any, Callable

import torch

from flame.module import Module


class LossBase(Module):
    def __init__(self, output_transform: Callable=lambda x: x) -> None:
        super(LossBase, self).__init__()
        self.output_transform = output_transform

    def _init(self):
        pass

    def forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any) -> Any:
        params = self.output_transform(args)
        return self.forward(*params)
