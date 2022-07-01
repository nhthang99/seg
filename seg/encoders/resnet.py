from copy import deepcopy

import torch.nn as nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter


class Resnet(nn.Module):
    def __init__(self, backbone_name, out_channels, pretrained=False, **kwargs) -> None:
        super(Resnet, self).__init__()
        assert backbone_name in resnet_encoders.keys()
        return_layers = {"bn1": "out_1", "layer1": "out_2", "layer2": "out_3", "layer3": "out_4", "layer4": "out_5"}
        backbone = resnet.__dict__[backbone_name](pretrained=pretrained, **kwargs)
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.backbone(x)


default_basic = {
    "encoder": Resnet,
    "params": {
        "out_channels": (64, 64, 128, 256, 512),
    }
}
default_bottleneck = {
    "encoder": Resnet,
    "params": {
        "out_channels": (64, 256, 512, 1024, 2048)
    }
}
resnet_encoders = {
    "resnet18": deepcopy(default_basic),
    "resnet34": deepcopy(default_basic),
    "resnet50": deepcopy(default_bottleneck),
    "resnet101": deepcopy(default_bottleneck),
    "resnet152": deepcopy(default_bottleneck),
    "resnext50_32x4d": deepcopy(default_bottleneck),
    "resnext101_32x8d": deepcopy(default_bottleneck)
}
