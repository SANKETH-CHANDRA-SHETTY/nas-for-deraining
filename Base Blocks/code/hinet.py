import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride
    )

class UNetConvBlock(nn.Module):
    """
    Basic 2-layer convolutional block with optional Half Instance Normalization (HIN).
    This version ensures out_channels == in_channels and no downsampling or upsampling,
    so input and output spatial size and channels are preserved.
    """
    def __init__(self, channels, relu_slope=0.2, use_HIN=True):
        super().__init__()
        self.identity = nn.Conv2d(channels, channels, 1, 1, 0)
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.use_HIN = use_HIN
        if use_HIN:
            # Apply InstanceNorm2d to half of the channels (even sized)
            if channels % 2 == 0:
                self.norm = nn.InstanceNorm2d(channels // 2, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(channels, affine=True)
    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            if out.shape[1] % 2 == 0:
                out_1, out_2 = torch.chunk(out, 2, dim=1)
                out = torch.cat([self.norm(out_1), out_2], dim=1)
            else:
                out = self.norm(out)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out

class HINet(nn.Module):
    """
    HINet encoder block with out_channels = in_channels and same spatial size.
    Use as a block in a UNet or NAS framework.
    """
    def __init__(self, in_channels=3,dummy=0):
        super().__init__()
        self.isOk = True
        self.block = UNetConvBlock(in_channels, relu_slope=0.2, use_HIN=True)
    def forward(self, x):
        return self.block(x)