from typing import Union

import torch.nn as nn

from seg.encoders.resnet import resnet_encoders
from seg.encoders.vgg import vgg_encoders
from seg.encoders.densenet import densenet_encoders
from seg.encoders.mobilenet import mobilenet_encoders


cfg_encoders = {}
cfg_encoders.update(resnet_encoders)
cfg_encoders.update(vgg_encoders)
cfg_encoders.update(densenet_encoders)
cfg_encoders.update(mobilenet_encoders)


class Encoder(nn.Module):
    def __init__(self, encoder: Union[str, nn.Module], pretrained=False, **kwargs) -> None:
        super(Encoder, self).__init__()
        if isinstance(encoder, str):
            self.encoder = self.get_encoder(encoder, pretrained)
        else:
            self.encoder = encoder
        assert hasattr(self.encoder, "out_channels"), \
               "Attribute `out_channels` does not defined in encoder"

    def get_encoder(self, encoder: str, pretrained=False) -> nn.Module:
        """Get default encoder

        Args:
            encoder (str): see default supported encoders at encoders/ folder
            pretrained (bool, optional): pretrained backbone. Defaults to False.

        Returns:
            nn.Module: encoder
        """
        assert cfg_encoders[encoder].get("encoder") is not None, \
               f"Encoder config should contain `encoder` attribute"
        encoder_class = cfg_encoders[encoder]["encoder"]
        params: dict = cfg_encoders[encoder].get("params", {})
        params.update(dict(backbone_name=encoder, pretrained=pretrained))
        return encoder_class(**params)

    @property
    def out_channels(self):
        return self.encoder.out_channels

    def forward(self, x):
        return tuple(self.encoder(x).values())
