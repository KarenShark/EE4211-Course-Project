import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms import CenterCrop

class UNet(nn.Module):
    """UNet model architecture"""

    def __init__(self, enc_channels: tuple, dec_channels: tuple, n_classes: int, output_size: tuple) -> None:
        super(UNet, self).__init__()
        # Store the output size
        self.output_size = output_size

        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.out = nn.Conv2d(dec_channels[-1], n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x[-1], x[::-1][1:])
        x = self.out(x)
        x = F.interpolate(x, self.output_size)

        return x

class Encoder(nn.Module):
    """Encoder module"""

    def __init__(self, channels: tuple) -> None:
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> list:

        outputs = list()

        for block in self.blocks:
            x = block(x)
            outputs.append(x)
            x = self.maxpool(x)

        return outputs


class Decoder(nn.Module):
    """Decoder module"""

    def __init__(self, channels: tuple) -> None:
        super(Decoder, self).__init__()
        self.n_channels = len(channels)
        self.blocks = nn.ModuleList([ConvBlock(channels[i], channels[i + 1])
                                     for i in range(len(channels) - 1)])
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
             for i in range(len(channels) - 1)])

    def forward(self, x: Tensor, enc_outputs: list) -> Tensor:
        for i in range(self.n_channels - 1):
            x = self.upconvs[i](x)
            # Resize the output from the skip connection to match a size of the up-scaling layer's output
            skip = CenterCrop(x.shape[2:])(enc_outputs[i])
            x = torch.concat([x, skip], dim=1)
            x = self.blocks[i](x)

        return x

class ConvBlock(nn.Module):
    """ Convolutional Block module"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            # 1st convolutional layer: (in_channels, H, W) => (out_channels, H, W)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 2nd convolutional layer: (out_channels, H, W) => (out_channels, H, W)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
