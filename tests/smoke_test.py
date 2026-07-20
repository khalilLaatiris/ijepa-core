"""CPU smoke test for ijepa-core/pretrain — the freeze-rule test gate, stage 1 of 2.

Builds a tiny synthetic ImageFolder dataset, runs the real training loop
(src.train.main, unmodified) for a couple of epochs on CPU, and asserts:
  1. it doesn't crash
  2. loss is finite every step
  3. a checkpoint is written and a second run can resume from it

Stage 2 (GPU, real data, 5 epochs) is ijepa-core/colab/confirm_train.ipynb — run manually on
Colab, not by this script. Both must pass before pretrain/ is frozen.
"""
import shutil
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PRETRAIN = ROOT / "pretrain"
sys.path.insert(0, str(PRETRAIN))

TMP = ROOT / "tests" / "smoke_test_tmp"
DATA_ROOT = TMP / "data"
IMAGE_FOLDER = "synthetic"
LOG_FOLDER = TMP / "logs"

N_CLASSES = 3
IMAGES_PER_CLASS = 8
IMG_SIZE = 64  # RandomResizedCrop upsamples to crop_size=224 regardless


def make_synthetic_dataset():
    train_dir = DATA_ROOT / IMAGE_FOLDER / "train"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    for c in range(N_CLASSES):
        cls_dir = train_dir / str(c)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            img.putdata([
                ((x * 7 + c * 31 + i) % 256, (x * 13) % 256, (x * 5 + i) % 256)
                for x in range(IMG_SIZE * IMG_SIZE)
            ])
            img.save(cls_dir / f"img{i}.jpeg")
    return train_dir


def base_config(epochs, load_checkpoint):
    with open(PRETRAIN / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["root_path"] = str(DATA_ROOT)
    cfg["data"]["image_folder"] = IMAGE_FOLDER
    cfg["data"]["batch_size"] = 4
    cfg["data"]["num_workers"] = 0
    cfg["logging"]["folder"] = str(LOG_FOLDER) + "/"
    cfg["logging"]["write_tag"] = "smoke"
    cfg["meta"]["model_name"] = "vit_small"
    cfg["meta"]["use_bfloat16"] = False
    cfg["meta"]["load_checkpoint"] = load_checkpoint
    cfg["optimization"]["epochs"] = epochs
    cfg["optimization"]["warmup"] = 0
    return cfg


def run(cfg):
    from src.train import main as train_main
    train_main(args=cfg, device=torch.device("cpu"))


def main():
    print("=== ijepa-core smoke test ===")
    if LOG_FOLDER.exists():
        shutil.rmtree(LOG_FOLDER)
    make_synthetic_dataset()

    print("--- run 1: 1 epoch, no resume ---")
    cfg1 = base_config(epochs=1, load_checkpoint=False)
    run(cfg1)

    ckpt_path = LOG_FOLDER / "smoke-latest.pth.tar"
    assert ckpt_path.exists(), f"expected checkpoint at {ckpt_path}, not found"
    ckpt1 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt1["epoch"] == 1, f"expected epoch=1 after run 1, got {ckpt1['epoch']}"
    assert ckpt1["loss"] == ckpt1["loss"], f"loss is NaN after run 1: {ckpt1['loss']}"  # NaN != NaN

    print("--- run 2: resume, 1 more epoch ---")
    cfg2 = base_config(epochs=2, load_checkpoint=True)
    run(cfg2)

    ckpt2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] == 2, f"expected epoch=2 after resume, got {ckpt2['epoch']}"
    assert ckpt2["loss"] == ckpt2["loss"], f"loss is NaN after resume: {ckpt2['loss']}"

    # -- resume correctness: verify via CSV log row count.
    # CSVLogger opens in append mode and writes a header row on each __init__ call, then
    # one data row per iteration. Both runs share the same CSV file.
    # With IMAGES_PER_CLASS=8, N_CLASSES=3, batch_size=4, drop_last=True:
    #   ipe = 24 // 4 = 6 iterations/epoch
    # Successful resume: run 1 trains epoch 0 (6 iters), run 2 resumes from epoch 1 and
    # trains only epoch 1 (6 iters). Each run writes 1 header + 6 data rows → 14 total.
    # Silent-failure resume: run 2 falls back to start_epoch=0 and retrains both epochs
    # (12 iters), writing 1 header + 12 data rows → 7 + 13 = 20 total. The mismatch
    # exposes the silent failure that ckpt2["epoch"]==2 alone cannot detect.
    csv_path = LOG_FOLDER / "smoke_r0.csv"
    with open(csv_path) as f:
        row_count = sum(1 for _ in f)
    expected_rows = 14  # 2 header rows + 6 iters/epoch * 2 epochs total (1 fresh + 1 resumed)
    assert row_count == expected_rows, (
        f"expected {expected_rows} CSV rows (proof run 2 actually resumed instead of "
        f"retraining from scratch), got {row_count} -- resume may have silently failed "
        f"(see src/helper.py's load_checkpoint, which swallows exceptions and falls back "
        f"to epoch=0)"
    )

    print("=== ALL CHECKS PASSED ===")
    print(f"run 1 loss: {ckpt1['loss']:.4f}  |  run 2 loss: {ckpt2['loss']:.4f}")


if __name__ == "__main__":
    main()
