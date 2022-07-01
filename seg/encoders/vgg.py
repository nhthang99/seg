from copy import deepcopy

import torch.nn as nn
from torchvision.models import vgg
from torchvision.models._utils import IntermediateLayerGetter


class VGG(nn.Module):
    def __init__(self, backbone_name, out_channels, pretrained=False, **kwargs) -> None:
        super(VGG, self).__init__()
        assert backbone_name in vgg_encoders.keys()
        backbone = vgg.__dict__[backbone_name](pretrained=pretrained, **kwargs)
        return_layers = self.get_return_layers(backbone)
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        self.out_channels = out_channels

    def get_return_layers(self, backbone):
        pool_idx = 0
        return_layers = {}
        for name, module in backbone.features.named_modules():
            # Get feature map at RELU layer before pooling layer
            if isinstance(module, nn.MaxPool2d):
                if 1 <= pool_idx <= 3:
                    return_layers.update({str(int(name) - 1): f"out_{pool_idx}"})
                elif pool_idx == 4:
                    return_layers.update({str(int(name) - 1): f"out_{pool_idx}"})
                    return_layers.update({name: f"out_{pool_idx + 1}"})
                pool_idx += 1

        return return_layers

    def forward(self, x):
        return self.backbone(x)


default_vgg = {
    "encoder": VGG,
    "params": {
        "out_channels": (128, 256, 512, 512, 512)
    }
}
vgg_encoders = {
    "vgg11": deepcopy(default_vgg),
    "vgg11_bn": deepcopy(default_vgg),
    "vgg13": deepcopy(default_vgg),
    "vgg13_bn": deepcopy(default_vgg),
    "vgg16": deepcopy(default_vgg),
    "vgg16_bn": deepcopy(default_vgg),
    "vgg19": deepcopy(default_vgg),
    "vgg19_bn": deepcopy(default_vgg),
}
