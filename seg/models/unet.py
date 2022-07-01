from typing import Union

from seg.decoders import Decoder
from seg.models.base import ModelBase


class UNet(ModelBase):
    def __init__(self, num_classes: int, encoder: str, decoder: Union[str, Decoder],
                 pretrained=False, activation=None, **kwargs):
        super(UNet, self).__init__(num_classes, encoder, decoder,
                                        pretrained=pretrained, activation=activation, **kwargs)
