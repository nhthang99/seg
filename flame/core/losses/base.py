from typing import Any, Callable

from flame.module import ModuleBase


class LossBase(ModuleBase):
    def __init__(self, output_transform: Callable) -> None:
        super(LossBase, self).__init__()
        self.output_transform = output_transform

    def _init(self):
        pass

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args: Any) -> Any:
        params = self.output_transform(args)
        return self.forward(*params)
