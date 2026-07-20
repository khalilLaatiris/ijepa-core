"""Segmentation decoder zoo — pluggable heads over a frozen encoder's tokens/feature pyramid.

Ported from Code/cswin_jepa_ret_pipeline_final_from_literature.ipynb (cell 3d). Trimmed to
ViT-only: drops the CSWin forward_pyramid branch and the model_name dispatch in
get_multiscale_feats -- this repo has no CSWin arm (see pretrain/'s own ViT-only scope), so ViT's
depth-tap pyramid is the only path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from downstream.common import CROP


class LinearHead(nn.Module):
    """Segmenter-linear / SETR-Naive: 1x1 conv + one-shot bilinear upsample. Cheapest baseline."""
    def __init__(self, in_ch, grid, n_classes, out_size=CROP, **_):
        super().__init__()
        self.out_size = out_size
        self.head = nn.Conv2d(in_ch, n_classes, 1)

    def forward(self, tokens):
        B, N, C = tokens.shape
        g = int(round(N ** 0.5))
        x = tokens.transpose(1, 2).reshape(B, C, g, g)
        return F.interpolate(self.head(x), size=self.out_size, mode="bilinear", align_corners=False)


class PUPDecoder(nn.Module):
    """SETR-PUP: progressive conv + 2x-upsample stack, 14 -> 224."""
    def __init__(self, in_ch, grid, n_classes, hidden=256, out_size=CROP, **_):
        super().__init__()
        self.grid, self.out_size = grid, out_size

        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

        self.dec = nn.Sequential(blk(in_ch, hidden), blk(hidden, hidden // 2),
                                  blk(hidden // 2, hidden // 4), blk(hidden // 4, hidden // 8))
        self.head = nn.Conv2d(hidden // 8, n_classes, 1)

    def forward(self, tokens):
        B, N, C = tokens.shape
        x = self.dec(tokens.transpose(1, 2).reshape(B, C, self.grid, self.grid))
        return self.head(F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False))


class FPNDecoder(nn.Module):
    """UPerNet/FPN: 1x1 laterals -> top-down upsample+add -> 3x3 smooth -> 1x1 classifier."""
    def __init__(self, in_chs, n_classes, hidden=256, out_size=CROP, **_):
        super().__init__()
        self.out_size = out_size
        self.laterals = nn.ModuleList([nn.Conv2d(c, hidden, 1) for c in in_chs])
        self.smooths = nn.ModuleList([nn.Conv2d(hidden, hidden, 3, padding=1) for _ in in_chs])
        self.head = nn.Conv2d(hidden, n_classes, 1)

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.laterals, feats)]
        x = lat[-1]
        fused = self.smooths[-1](x)
        for i in range(len(lat) - 2, -1, -1):
            x = F.interpolate(x, size=lat[i].shape[-2:], mode="bilinear", align_corners=False) + lat[i]
            fused = self.smooths[i](x)
        return self.head(F.interpolate(fused, size=self.out_size, mode="bilinear", align_corners=False))


def get_multiscale_feats(enc, imgs, n_taps=3):
    """Depth-tap pyramid (DPT, Ranftl et al. 2021) -- ViT has no native multiscale pyramid, so this
    stands in for one by tapping n_taps evenly-spaced transformer blocks. All taps are at the SAME
    14x14 resolution (ViT blocks don't downsample), so FPNDecoder's top-down interpolate+add is a
    structural no-op here -- kept anyway so all three decoders share one interface."""
    blocks, n = enc.blocks, len(enc.blocks)
    idxs = sorted({max(0, round((i + 1) * n / n_taps) - 1) for i in range(n_taps)})
    taps = {}

    def mk(i):
        def hook(_, __, out):
            taps[i] = out
        return hook

    handles = [blocks[i].register_forward_hook(mk(i)) for i in idxs]
    try:
        enc(imgs)
    finally:
        for h in handles:
            h.remove()
    feats = []
    for i in idxs:
        x = taps[i]
        B, N, C = x.shape
        g = int(round(N ** 0.5))
        feats.append(x.transpose(1, 2).reshape(B, C, g, g).contiguous())
    return feats


DECODER_ZOO = {"linear": LinearHead, "pup": PUPDecoder}
SEG_DECODERS = ("linear", "pup", "fpn")


def build_decoder(kind, enc, C, grid, n_classes, device):
    if kind == "fpn":
        with torch.no_grad():
            in_chs = [f.shape[1] for f in
                      get_multiscale_feats(enc, torch.zeros(1, 3, CROP, CROP, device=device))]
        return FPNDecoder(in_chs, n_classes).to(device), dict(cls="FPNDecoder", in_chs=in_chs, n_classes=n_classes)
    arch = dict(cls=DECODER_ZOO[kind].__name__, in_ch=int(C), grid=int(grid), n_classes=n_classes)
    return DECODER_ZOO[kind](C, grid, n_classes).to(device), arch
