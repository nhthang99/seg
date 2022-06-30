import torch.nn as nn
from torchvision.models import densenet
from torchvision.models._utils import IntermediateLayerGetter


class DenseNet(nn.Module):
    def __init__(self, backbone_name, out_channels, pretrained=False, **kwargs) -> None:
        super(DenseNet, self).__init__()
        assert backbone_name in densenet_encoders.keys()
        backbone = densenet.__dict__[backbone_name](pretrained=pretrained, **kwargs)
        return_layers = {"transition1": "1", "transition2": "2", "transition3": "3", "norm5": "4"}
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.backbone(x)

densenet_encoders = {
    "densenet121": {
        "encoder": DenseNet,
        "params": {
            "out_channels": (256, 512, 1024, 1024),
        },
    },
    "densenet169": {
        "encoder": DenseNet,
        "params": {
            "out_channels": (256, 512, 1280, 1664),
        },
    },
    "densenet201": {
        "encoder": DenseNet,
        "params": {
            "out_channels": (256, 512, 1792, 1920),
        },
    },
    "densenet161": {
        "encoder": DenseNet,
        "params": {
            "out_channels": (384, 768, 2112, 2208),
        },
    },
}
