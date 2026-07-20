"""Shared downstream-eval building blocks: encoder loading, transforms, metrics.

Ported from Code/cswin_jepa_ret_pipeline_final_from_literature.ipynb (cell 0c "Shared constants"
and cell 3b "Frozen-encoder loader") -- read-only reference, never modified. Trimmed to ViT-only:
the source notebook juggled two vendored repos (ijepa-main / ijepa-cswin) via sys.path swapping to
compare ViT against CSWin; this repo has no CSWin arm, so load_frozen_encoder imports directly from
ijepa-core's own pretrain/ package instead.
"""
import sys
from pathlib import Path

import numpy as np
import torch

PRETRAIN_DIR = Path(__file__).resolve().parent.parent / "pretrain"

CROP = 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# FOV masks: DRIVE ships them; STARE ships none, so the field generates one (Orlando et al.,
# SIPAIM 2016). Threshold validated against DRIVE's SHIPPED mask: IoU 0.996 at 0.19.
FOV_THRESHOLD = 0.19

# The retinal-vessel literature reports a TUPLE with AUC-ROC as the headline, not Dice alone.
# Needs SOFT probabilities (argmax destroys the AUCs) and is FOV-restricted (score only inside the
# retinal field of view -- background outside it is trivial and inflates Acc/Spec).
VESSEL_THRESHOLD = 0.5
VESSEL_COLS = ["auc_roc", "auc_pr", "f1", "sens", "spec", "acc", "iou"]


# Classification: the ImageNet centre-crop convention. Correct for a whole-image label.
def eval_transform(size=CROP):
    import torchvision.transforms as T
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size),
                       T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# Segmentation: ONE square geometry for image, mask AND FOV. Do not swap this for eval_transform
# (which centre-crops) -- that mismatch measured IoU 0.071 between image and mask geometry, i.e.
# they never lined up and vessel Dice was scoring misregistration, not the model.
def seg_transform(size=CROP):
    import torchvision.transforms as T
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
                       T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def resize_label(pil_img, size=CROP):
    """Masks / FOV maps: NEAREST at the SAME square geometry seg_transform gives the image."""
    from PIL import Image as _I
    return pil_img.resize((size, size), _I.NEAREST)


def get_fov_mask(image_rgb, threshold=FOV_THRESHOLD):
    from skimage import color, measure
    from scipy import ndimage, stats
    lab = color.rgb2lab(image_rgb)
    lab[:, :, 0] /= 100.0
    mask = ndimage.binary_fill_holes(lab[:, :, 0] >= threshold)
    mask = ndimage.median_filter(mask, size=(5, 5))
    cc = measure.label(mask).astype(float)
    cc[cc == mask[0][0]] = np.nan
    return cc == stats.mode(cc, axis=None, nan_policy="omit")[0]


def denorm(x):
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def dice_per_class(logits, target, n_classes, eps=1e-6):
    pred = logits.argmax(1)
    out = []
    for k in range(n_classes):
        p, t = (pred == k), (target == k)
        denom = p.sum().item() + t.sum().item()
        out.append((2 * (p & t).sum().item() + eps) / (denom + eps) if denom > 0 else float("nan"))
    return out


def vessel_metrics(prob, gt, fov=None, eps=1e-9):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as _auc
    prob = np.asarray(prob, dtype=np.float64).ravel()
    gt = np.asarray(gt).ravel().astype(bool)
    if fov is not None:
        keep = np.asarray(fov).ravel().astype(bool)
        prob, gt = prob[keep], gt[keep]
    pred = prob >= VESSEL_THRESHOLD
    tp = int((pred & gt).sum()); tn = int((~pred & ~gt).sum())
    fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    precision = tp / (tp + fp + eps)
    sen = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * precision * sen / (precision + sen + eps)  # == Dice for binary
    iou = tp / (tp + fp + fn + eps)
    if gt.any() and (~gt).any():
        roc = float(roc_auc_score(gt, prob))
        pre, rec, _ = precision_recall_curve(gt, prob)
        pr = float(_auc(rec, pre))
    else:
        roc = pr = float("nan")  # single-class region -> NaN, never a silent 0.5
    return dict(auc_roc=roc, auc_pr=pr, acc=float(acc), sens=float(sen), spec=float(spec),
                f1=float(f1), precision=float(precision), iou=float(iou),
                vessel_frac=float(gt.mean()), n_px=int(gt.size))


def load_frozen_encoder(ckpt_path, model_name="vit_base", device=None):
    """Loads the EMA target_encoder -- I-JEPA's evaluation convention, not the context encoder."""
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(PRETRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(PRETRAIN_DIR))
    from src.helper import init_model

    encoder, _ = init_model(device=device, patch_size=16, model_name=model_name,
                             crop_size=CROP, pred_depth=6, pred_emb_dim=384)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # self-produced (RNG state, optimizer)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ck["target_encoder"].items()}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    # Hard-fail: a silent partial load runs the encoder on RANDOM weights and yields
    # plausible-looking garbage metrics -- the worst possible failure mode here.
    assert not missing and not unexpected, (
        f"ckpt key mismatch for {model_name}: {len(missing)} missing, {len(unexpected)} unexpected")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    dim = encoder.embed_dim if hasattr(encoder, "embed_dim") else encoder.norm.weight.shape[0]
    grid = int(round((encoder.num_patches if hasattr(encoder, "num_patches")
                       else encoder.patch_embed.num_patches) ** 0.5))
    return encoder, dim, grid


@torch.no_grad()
def extract_tokens(enc, x):
    return enc(x)


@torch.no_grad()
def extract_pooled(enc, x):
    return enc(x).mean(dim=1)
