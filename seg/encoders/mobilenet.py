import torch.nn as nn
from torchvision.models import mobilenet
from torchvision.models._utils import IntermediateLayerGetter


class MobileNet(nn.Module):
    def __init__(self, backbone_name, out_channels, pretrained=False, **kwargs) -> None:
        super(MobileNet, self).__init__()
        assert backbone_name in mobilenet_encoders.keys()
        backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, **kwargs)
        return_layers = {"3": "1", "6": "2", "13": "3", "18": "4"}
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.backbone(x)


mobilenet_encoders = {
    "mobilenet_v2": {
        "encoder": MobileNet,
        "params": {
            "out_channels": (24, 32, 96, 1280)
        }
    }
}
