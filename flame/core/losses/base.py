from typing import Any, Callable, Optional

import torch

from flame.module import Module


class LossBase(Module):
    def __init__(self, mode, num_classes: Optional[int]=None,
                 output_transform: Callable=lambda x: x) -> None:
        super(LossBase, self).__init__()
        assert mode in ["binary", "multi_class", "multi_label"]
        self.mode = mode
        if mode == "multi_class" and not isinstance(num_classes, int):
            raise AttributeError("Attribute `num_classes` should be integer during multi-class mode "
                                 f"instead of {type(num_classes)}")
        self.num_classes = num_classes # Only used in multi-class mode
        self.output_transform = output_transform

    def init(self):
        pass

    def forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any) -> Any:
        params = self.output_transform(args)
        return self.forward(*params)
