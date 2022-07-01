import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=None) -> None:
        super(UNetDecoder, self).__init__()
        assert len(encoder_channels), "Encoder channels is empty"
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]

        if decoder_channels is None:
            decoder_channels = tuple([max(16, encoder_channels[0] // 2**i)
                                      for i in range(1, len(encoder_channels))])

        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_chs, skip_chs, out_chs)
            for in_chs, skip_chs, out_chs in zip(in_channels, skip_channels, out_channels)
        ])
        self.out_channels = decoder_channels

    def forward(self, features):
        features = features[::-1] # Reverse feature map with start from head of encoder
        x, *skips = features

        for idx, decoder_block in enumerate(self.decoder_blocks):
            skip = skips[idx] if idx < len(skips) else None
            x = decoder_block(x, skip)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels,
                                kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels,
                                kernel_size=3, padding=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding=0, stride=1):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=False)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


unet_decoders = {
    "unet_decoder": {
        "decoder": UNetDecoder,
        "params": {
            "decoder_channels": (256, 128, 64, 32, 16)
        }
    }
}
