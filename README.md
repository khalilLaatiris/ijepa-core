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

## Private repo — access setup (one-time, per machine)

This repo is **private**. Anonymous `git clone` fails everywhere — HPC-MARWAN and Colab each need a
one-time credential setup before their first clone.

### HPC-MARWAN (login node) — SSH deploy key

A shared cluster's shell history and `.git/config` are the wrong place for a token, so this uses an
SSH deploy key instead (read-only, scoped to just this repo).

```bash
# 1. Generate a dedicated keypair (once) — do NOT reuse your personal GitHub SSH key here.
ssh-keygen -t ed25519 -f ~/.ssh/ijepa_core_deploy -N "" -C "hpc-marwan-ijepa-core"

# 2. Print the PUBLIC key and add it on GitHub:
#      github.com/khalilLaatiris/ijepa-core -> Settings -> Deploy keys -> Add deploy key
#      (leave "Allow write access" UNCHECKED -- read-only is all a clone/pull needs)
cat ~/.ssh/ijepa_core_deploy.pub

# 3. Point SSH at this key for github.com (add to ~/.ssh/config):
cat >> ~/.ssh/config <<'EOF'
Host github-ijepa-core
    HostName github.com
    User git
    IdentityFile ~/.ssh/ijepa_core_deploy
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

# 4. Clone using the alias:
git clone github-ijepa-core:khalilLaatiris/ijepa-core.git /home/$USER/retina/repos/ijepa-core
```

Later updates: `cd /home/$USER/retina/repos/ijepa-core && git pull` — the alias is already wired
into that clone's remote, no need to repeat the URL.

### Colab — repo-scoped fine-grained PAT via Colab Secret

Same pattern the sibling project uses for `KAGGLE_API_TOKEN` (see
`+ shaped JEPA/notebooks/phase1_stratified_subset_colab.ipynb`).

1. Create a **fine-grained** PAT at `github.com/settings/tokens?type=beta`, scoped to **only**
   `khalilLaatiris/ijepa-core`, **read-only** (Contents: Read-only is enough).
2. In Colab: left sidebar → 🔑 Secrets → add secret named `GITHUB_PAT`, paste the token, enable
   "Notebook access" for `confirm_train.ipynb`.
3. `colab/confirm_train.ipynb`'s clone cell reads it via `google.colab.userdata.get("GITHUB_PAT")`
   — never printed, never written to Drive, never appears in a traceback (clone/pull failures are
   caught and re-raised with the token stripped from the error message).

## HPC-MARWAN — training jobs

Once cloned (above), `../Code/`'s SLURM jobs invoke this engine — see `Code/CLAUDE.md`.

## Colab — confirm-train

`colab/confirm_train.ipynb` mounts Drive and clones this repo to
`/content/drive/MyDrive/ijepa-core/` — that Drive copy is a convenience clone for session
persistence, not a separate distribution channel. GitHub is the single source of truth.
