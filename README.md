# ijepa-core

Canonical, single-GPU I-JEPA pretraining engine for the retina-diseases research thread —
ViT-S/ViT-B, same masking/EMA/loss/schedule math as `facebookresearch/ijepa`'s `main.py`/
`src/train.py`, vendored-unmodified upstream model/mask/scheduler code, DDP dropped (single GPU
only). Replaces the earlier pristine-clone approach (`Code/`'s old `slurm/train/ijepa.sl`), which
had no confirmed-working single-GPU run on either Colab or HPC-MARWAN.

Used by `../Code/` (HPC-MARWAN training jobs) and available to any sibling subproject under this
root. See `CLAUDE.md` for the freeze rule — read it before editing `pretrain/`.

## Usage

`pretrain/` only trains — it does not stage a dataset. Point it at any ImageFolder-shaped tree with
a `train/<class>/<img>` level:

```bash
cd pretrain
python main.py --fname config.yaml --device cuda:0   # or --device cpu
```

Resume: set `meta.load_checkpoint: true` in the config — picks up `<write_tag>-latest.pth.tar`
from `logging.folder`.

## What's vendored unmodified vs. adapted

| File | Status |
|---|---|
| `src/models/vision_transformer.py` | vendored unmodified (ViT encoder + predictor) |
| `src/masks/multiblock.py`, `src/masks/utils.py` | vendored unmodified |
| `src/utils/schedulers.py`, `src/utils/tensors.py`, `src/utils/logging.py` | vendored unmodified |
| `src/transforms.py` | vendored unmodified |
| `src/datasets/imagenet1k.py` | vendored unmodified (plain `ImageFolder` wrapper, needs `train/`) |
| `src/helper.py` | vendored unmodified |
| `src/train.py` | adapted — DDP/`init_distributed`/`AllReduce` dropped (no-ops at world_size=1 anyway); `DistributedSampler` still built (`world_size=1, rank=0`) since the dataset loader requires it |
| `main.py` | new — single-device CLI |

## Test gate (both required before the freeze rule applies)

1. `tests/smoke_test.py` — CPU, synthetic data, run locally. Passing as of the commit that added it.
2. `colab/confirm_train.ipynb` — GPU, real stratified-subset data, 5 epochs, ViT-S. Run manually on
   Colab (this repo has no CI/GPU access). Not yet run — see `CLAUDE.md`.

## HPC-MARWAN

Clone directly to `/home/$USER/retina/repos/ijepa-core` on the login node (compute nodes have no
internet, same pattern the old pristine-clone path used). `../Code/`'s SLURM jobs invoke this
engine — see `Code/CLAUDE.md`.

## Colab

`colab/confirm_train.ipynb` mounts Drive and clones this repo to
`/content/drive/MyDrive/ijepa-core/` — that Drive copy is a convenience clone for session
persistence, not a separate distribution channel. GitHub is the single source of truth.
