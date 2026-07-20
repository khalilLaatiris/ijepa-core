"""Tests for downstream/common.py — metrics correctness against known-good synthetic cases."""
import numpy as np
import torch

from downstream.common import dice_per_class, vessel_metrics, seg_transform, eval_transform, CROP


def test_dice_per_class_perfect_match():
    # logits that argmax to exactly `target` for both classes -> Dice 1.0 everywhere.
    target = torch.tensor([[[0, 1], [1, 0]]])  # (B=1, H=2, W=2)
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 0] = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])  # class 0 wins where target==0
    logits[0, 1] = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])  # class 1 wins where target==1
    dice = dice_per_class(logits, target, n_classes=2)
    assert dice == [1.0, 1.0]


def test_dice_per_class_class_absent_is_nan_not_zero():
    # class 1 never appears in target or prediction -> NaN (unmeasured), not 0.0.
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 0] = 10.0  # class 0 always wins
    dice = dice_per_class(logits, target, n_classes=2)
    assert dice[0] == 1.0
    assert dice[1] != dice[1]  # NaN != NaN


def test_vessel_metrics_separable_gives_auc_near_one():
    rng = np.random.default_rng(0)
    n = 2000
    gt = rng.random(n) > 0.9  # ~10% positive, matches real vessel fraction
    # separable: positives get high prob, negatives get low prob, small noise
    prob = np.where(gt, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n))
    m = vessel_metrics(prob, gt)
    assert m["auc_roc"] > 0.95
    assert m["auc_pr"] > 0.8
    assert m["n_px"] == n
    assert abs(m["vessel_frac"] - gt.mean()) < 1e-9


def test_vessel_metrics_single_class_region_is_nan_not_default():
    # every pixel background -> AUC undefined -> NaN, never a silent 0.5.
    gt = np.zeros(100, dtype=bool)
    prob = np.random.default_rng(0).random(100)
    m = vessel_metrics(prob, gt)
    assert m["auc_roc"] != m["auc_roc"]  # NaN
    assert m["auc_pr"] != m["auc_pr"]    # NaN


def test_vessel_metrics_fov_restriction_excludes_pixels():
    gt = np.array([True, True, False, False])
    prob = np.array([0.9, 0.9, 0.9, 0.9])  # all high -> without FOV, 2 false positives
    fov = np.array([True, True, True, False])  # last pixel excluded
    m = vessel_metrics(prob, gt, fov=fov)
    assert m["n_px"] == 3  # one pixel dropped by the FOV mask


def test_seg_transform_and_eval_transform_produce_CROPxCROP():
    from PIL import Image
    img = Image.new("RGB", (100, 60))  # deliberately non-square, smaller than CROP
    out_seg = seg_transform()(img)
    out_eval = eval_transform()(img)
    assert out_seg.shape == (3, CROP, CROP)
    assert out_eval.shape == (3, CROP, CROP)
