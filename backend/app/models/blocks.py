"""Model blocks: ResBlock3D + StateSpaceBlock (Mamba-style).

[REPORT] hybrid CNN + state-space; residual learning; skip connections.
True Mamba is listed as *future work*, so StateSpaceBlock uses a real Mamba
if `mamba_ssm` is installed (Linux + CUDA), else a gated depthwise-conv
sequence mixer as a light stand-in. [CHOICE] channel widths live in net.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.InstanceNorm3d(cout, affine=True)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.InstanceNorm3d(cout, affine=True)
        self.skip = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride, bias=False),
                nn.InstanceNorm3d(cout, affine=True),
            )
        )

    def forward(self, x):
        y = F.leaky_relu(self.n1(self.c1(x)), 0.01)
        y = self.n2(self.c2(y))
        return F.leaky_relu(y + self.skip(x), 0.01)


class StateSpaceBlock(nn.Module):
    """Long-range context over the flattened 3D feature map."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        try:
            from mamba_ssm import Mamba

            self.mixer = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self.real = True
        except Exception:
            self.real = False
            self.in_proj = nn.Linear(dim, 2 * dim)
            self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
            self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):  # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        s = x.flatten(2).transpose(1, 2)  # [B, N, C]
        h = self.norm(s)
        if self.real:
            h = self.mixer(h)
        else:
            a, g = self.in_proj(h).chunk(2, -1)
            a = self.dw(a.transpose(1, 2)).transpose(1, 2)
            h = self.out_proj(F.silu(a) * torch.sigmoid(g))
        return (s + h).transpose(1, 2).reshape(B, C, D, H, W)
