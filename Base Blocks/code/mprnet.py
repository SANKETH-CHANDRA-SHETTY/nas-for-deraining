import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=bias),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=False, act=None):
        super(CAB, self).__init__()
        if act is None:
            act = nn.PReLU()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias)
        )
        # Pass reduction to CALayer
        self.CA = CALayer(n_feat, reduction, bias=bias)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class MPRNet(nn.Module):
    """
    Single encoder block from MPRNet to be used as a UNet block for NAS.
    - out_channels = in_channels
    - output H, W same as input
    - No downsampling, no skip connection, no for-loop repeat
    """
    def __init__(self, in_channels=3,dummy=0):
        super().__init__()
        self.isOk = True
        self.block = CAB(in_channels, kernel_size=3, reduction=3, bias=False, act=nn.PReLU())
    def forward(self, x):
        return self.block(x)