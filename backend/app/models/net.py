"""TriViewResMambaUNet: encoder-decoder with skip connections + deep supervision."""
import torch
import torch.nn as nn

from .blocks import ResBlock3D, StateSpaceBlock


class TriViewResMambaUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16, deep_supervision=True):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.ds = deep_supervision
        self.enc = nn.ModuleList(
            [
                ResBlock3D(in_ch, c[0]),
                ResBlock3D(c[0], c[1], 2),
                ResBlock3D(c[1], c[2], 2),
                ResBlock3D(c[2], c[3], 2),
                ResBlock3D(c[3], c[4], 2),
            ]
        )
        self.ssm = StateSpaceBlock(c[4])
        self.up = nn.ModuleList([nn.ConvTranspose3d(c[i + 1], c[i], 2, 2) for i in range(4)])
        self.dec = nn.ModuleList([ResBlock3D(c[i] * 2, c[i]) for i in range(4)])
        self.head = nn.Conv3d(c[0], out_ch, 1)
        self.aux = nn.ModuleList([nn.Conv3d(c[i], out_ch, 1) for i in (1, 2, 3)])

    def forward(self, x):
        feats = []
        for e in self.enc:
            x = e(x)
            feats.append(x)
        x = self.ssm(feats[-1])
        aux_out = []
        for i in (3, 2, 1, 0):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, feats[i]], 1))
            if self.ds and self.training and i in (3, 2, 1):
                aux_out.append(self.aux[i - 1](x))
        out = self.head(x)
        return (out, aux_out) if (self.ds and self.training) else out
