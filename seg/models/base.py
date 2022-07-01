from typing import Union

import torch.nn as nn

from seg.encoders import Encoder
from seg.decoders import Decoder


class ModelBase(nn.Module):
    def __init__(self, num_classes: int, encoder: Union[str, Encoder],
                 decoder: Union[str, Decoder], neck=None,
                 pretrained=False, activation=None, **kwargs):
        super(ModelBase, self).__init__()
        self.encoder = Encoder(encoder, pretrained=pretrained)
        self.neck = neck # Does not supported now
        self.decoder = Decoder(decoder, self.encoder.out_channels, **kwargs)

        self.seg_head = SegmentationHead(self.decoder.out_channels[-1], num_classes)
        self.activation = Activation(activation)

    def forward(self, x):
        feats = self.encoder(x)
        if self.neck:
            feats = self.neck(feats)
        decoder_outs = self.decoder(feats)

        masks = self.seg_head(decoder_outs)
        masks = self.activation(masks)

        return masks


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {name}")

    def forward(self, x):
        return self.activation(x)
