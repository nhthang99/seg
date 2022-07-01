import torch.nn as nn
from torchvision.models import mobilenet
from torchvision.models._utils import IntermediateLayerGetter


class MobileNet(nn.Module):
    def __init__(self, backbone_name, out_channels, pretrained=False, **kwargs) -> None:
        super(MobileNet, self).__init__()
        assert backbone_name in mobilenet_encoders.keys()
        backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, **kwargs)
        return_layers = {"1": "out_1", "3": "out_2", "6": "out_3", "13": "out_4", "18": "out_5"}
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.backbone(x)


mobilenet_encoders = {
    "mobilenet_v2": {
        "encoder": MobileNet,
        "params": {
            "out_channels": (16, 24, 32, 96, 1280)
        }
    }
}
