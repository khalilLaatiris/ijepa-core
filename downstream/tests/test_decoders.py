"""Tests for downstream/decoders.py — output shapes and pyramid-tap extraction, CPU-only."""
import sys
from pathlib import Path

import torch

PRETRAIN_DIR = Path(__file__).resolve().parent.parent.parent / "pretrain"
if str(PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_DIR))

from downstream.common import CROP
from downstream.decoders import LinearHead, PUPDecoder, FPNDecoder, get_multiscale_feats, build_decoder, DECODER_ZOO


def _fake_tokens(batch=2, grid=14, dim=384):
    n = grid * grid
    return torch.randn(batch, n, dim)


def test_linear_head_output_shape():
    dec = LinearHead(in_ch=384, grid=14, n_classes=3)
    out = dec(_fake_tokens(dim=384))
    assert out.shape == (2, 3, CROP, CROP)


def test_pup_decoder_output_shape():
    dec = PUPDecoder(in_ch=384, grid=14, n_classes=2)
    out = dec(_fake_tokens(dim=384))
    assert out.shape == (2, 2, CROP, CROP)


def test_fpn_decoder_output_shape():
    # 3 feature maps at the same 14x14 resolution, differing channel counts (as get_multiscale_feats
    # would tap from real transformer blocks, all same spatial size for a ViT).
    feats = [torch.randn(2, c, 14, 14) for c in (384, 384, 384)]
    dec = FPNDecoder(in_chs=[384, 384, 384], n_classes=2)
    out = dec(feats)
    assert out.shape == (2, 2, CROP, CROP)


def test_get_multiscale_feats_taps_real_vit_small():
    from src.helper import init_model
    device = torch.device("cpu")
    encoder, _ = init_model(device=device, patch_size=16, model_name="vit_small",
                             crop_size=CROP, pred_depth=6, pred_emb_dim=384)
    encoder.eval()
    imgs = torch.randn(1, 3, CROP, CROP)
    with torch.no_grad():
        feats = get_multiscale_feats(encoder, imgs, n_taps=3)
    assert len(feats) == 3
    for f in feats:
        assert f.shape[0] == 1
        assert f.shape[2] == f.shape[3] == 14  # 224/16 patch grid, unchanged across ViT depth


def test_build_decoder_linear_and_fpn():
    from src.helper import init_model
    device = torch.device("cpu")
    encoder, _ = init_model(device=device, patch_size=16, model_name="vit_small",
                             crop_size=CROP, pred_depth=6, pred_emb_dim=384)
    encoder.eval()

    dec, arch = build_decoder("linear", encoder, C=384, grid=14, n_classes=2, device=device)
    assert isinstance(dec, DECODER_ZOO["linear"])
    assert arch["cls"] == "LinearHead"

    dec, arch = build_decoder("fpn", encoder, C=384, grid=14, n_classes=2, device=device)
    assert isinstance(dec, FPNDecoder)
    assert arch["cls"] == "FPNDecoder"
    assert len(arch["in_chs"]) == 3
