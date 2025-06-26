import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.prelu  = nn.PReLU()
        self.conv2  = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(channel)
        self.se     = SELayer(channel, reduction)
    def forward(self, x):
        residual = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.prelu(out)
        return out

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        # Only perform DWT if h and w >= 2, else return as is
        h, w = x.shape[2], x.shape[3]
        if h < 2 or w < 2:
            return x
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        # Concatenate and reshape back to original shape if possible
        out = torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
        return out

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel = int(in_channel / (r ** 2))
        # If in_channel is not divisible by 4, just return as is
        if in_channel % 4 != 0:
            return x
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        device = x.device
        out_height, out_width = r * in_height, r * in_width
        h = torch.zeros([in_batch, out_channel, out_height, out_width], dtype=x.dtype, device=device)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h

class MCWNet(nn.Module):
    """
    A single MCWNet encoder-like block, with out_channels = in_channels and output H*W same as input.
    No skip connection. Suitable for UNet/NAS plug-in.
    """
    def __init__(self, in_channels=3, dummy=0):
        super().__init__()
        reduction=16

        self.isOk = True

        self.se1 = SEBlock(in_channels, reduction)
        self.se2 = SEBlock(in_channels, reduction)
        self.se3 = SEBlock(in_channels, reduction)
    def forward(self, x):
        x = self.se1(x)
        x = self.se2(x)
        x = self.se3(x)
        return x