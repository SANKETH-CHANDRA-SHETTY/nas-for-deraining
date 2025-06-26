import torch
import torch.nn as nn
from einops import rearrange

# LayerNorm2d and LayerNormFunction as in your paste.txt
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.sg = SimpleGate()
    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, is_prompt=False, bias=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.is_prompt = is_prompt
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.prompt = nn.Parameter(torch.ones(num_heads, dim//num_heads, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def with_prompt(self, tensor, prompt):
        return tensor if prompt is None else tensor + prompt
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        if self.is_prompt:
            prompt = self.prompt
            q = self.with_prompt(q, prompt)
            k = self.with_prompt(k, prompt)
            v = self.with_prompt(v, prompt)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class CAPTNet(nn.Module):
    """
    CAPTNet transformer block for UNet/NAS with fixed dim=48 for attention/FFN.
    Input/output: [B, in_channels, H, W] -> [B, in_channels, H, W]
    num_heads: must divide 48 (e.g. 1, 2, 3, 4, 6, 8, 12, 16, 24, 48)
    """
    def __init__(self, in_channels=3, heads=4):
        super().__init__()
        dim = 48
        prompt=False

        self.isOk = True

        # Validate heads
        if (heads!=4 and heads != 8 and heads != 16 ) :
            print(f"Invalid heads: {heads}. Must be [4, 8, 16]")
            self.isOk = False
            return

        self.in_proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, heads, prompt)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim)
        self.out_proj = nn.Conv2d(dim, in_channels, kernel_size=1)
    def forward(self, x):
        if not self.isOk:
            print("Model not initialized correctly due to invalid heads.")
            return None
        x_proj = self.in_proj(x)
        x_proj = x_proj + self.attn(self.norm1(x_proj))
        x_proj = x_proj + self.ffn(self.norm2(x_proj))
        x_proj = self.out_proj(x_proj)
        return x_proj
