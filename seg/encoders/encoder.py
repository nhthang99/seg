import torch.nn as nn

from seg.encoders import cfg_encoders


class Encoder(nn.Module):
    def __init__(self, backbone_name, pretrained=False, **kwargs) -> None:
        super(Encoder, self).__init__()
        self.backbone_name = backbone_name
        self.init_encoder(backbone_name, pretrained)

    def init_encoder(self, backbone_name, pretrained=False):
        assert cfg_encoders[backbone_name].get("encoder") is not None, \
               f"Encoder config should contain `encoder` attribute"
        encoder_class = cfg_encoders[backbone_name]["encoder"]
        params: dict = cfg_encoders[backbone_name].get("params", {})
        params.update(dict(backbone_name=backbone_name, pretrained=pretrained))
        self.encoder = encoder_class(**params)

    def out_channels(self):
        return self.encoder.out_channels

    def forward(self, x):
        return self.encoder(x)
