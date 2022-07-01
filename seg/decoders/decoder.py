from typing import Union

import torch.nn as nn

from seg.decoders.unet import unet_decoders


cfg_decoders = {}
cfg_decoders.update(unet_decoders)


class Decoder(nn.Module):
    def __init__(self, decoder: Union[str, nn.Module], encoder_channels, **kwargs) -> None:
        super(Decoder, self).__init__()
        if isinstance(decoder, str):
            self.decoder = self.get_decoder(decoder, encoder_channels)
        else:
            self.decoder = decoder
        assert hasattr(self.decoder, "out_channels"), \
               "Attribute `out_channels` does not defined in encoder"

    def get_decoder(self, decoder: str, encoder_channels) -> None:
        """Get default decoder

        Args:
            encoder (str): see default supported encoders at encoders/ folder
            pretrained (bool, optional): pretrained backbone. Defaults to False.

        Returns:
            nn.Module: encoder
        """
        assert cfg_decoders[decoder].get("decoder") is not None, \
               f"Decoder config should contain `decoder` attribute"
        decoder_class = cfg_decoders[decoder]["decoder"]
        params: dict = cfg_decoders[decoder].get("params", {})
        params.update(dict(encoder_channels=encoder_channels))
        return decoder_class(**params)

    @property
    def out_channels(self):
        return self.decoder.out_channels

    def forward(self, x):
        return self.decoder(x)
