# ijepa-core

> **Fork notice.** This is a clone of Meta's official I-JEPA codebase —
> [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) ([paper](https://arxiv.org/pdf/2301.08243.pdf),
> CVPR-23) — with one fix applied: **single-GPU training was broken upstream** (`main.py` always
> spawned a worker process via `mp.Process` even for a single device, and `init_distributed()`
> tried to init a NCCL process group, which hangs waiting for peer ranks that never join). Fixed in
> two files:
> - `main.py` — single-device runs execute `process_main` directly instead of going through
>   `mp.Process`/`mp.set_start_method('spawn')`.
> - `src/utils/distributed.py` — `init_distributed()` returns `(1, 0)` immediately when
>   `world_size <= 1`, skipping `torch.distributed.init_process_group` entirely.
>
> Multi-GPU/SLURM training (`main_distributed.py`, `submitit`) is untouched upstream code. Everything
> else — model, masking, transforms, schedulers — is vendored unmodified.

Canonical I-JEPA training engine for the retina-diseases research thread. Used by `../Code/`'s
HPC-MARWAN SLURM jobs and available to any sibling subproject under the project root — see the root
`CLAUDE.md` there for the "always use this repo for I-JEPA training" rule.

## Method

I-JEPA (Image-based Joint-Embedding Predictive Architecture) is a self-supervised method that
predicts the representations of part of an image from the representations of other parts of the
same image, in latent space (no pixel-level reconstruction, no hand-crafted augmentation
invariances).

## Code structure

```
.
├── configs                   # experiment '.yaml' configs
├── src
│   ├── train.py              # the I-JEPA training loop
│   ├── helper.py              # model/opt init, checkpoint loading
│   ├── transforms.py          # pretrain data transforms
│   ├── datasets                # ImageFolder-style loaders
│   ├── models                  # ViT encoder + predictor
│   ├── masks                   # mask collators/utilities
│   └── utils                   # distributed, schedulers, tensors, logging
├── main_distributed.py       # SLURM/multi-GPU entrypoint (submitit)
└── main.py                   # single/multi-GPU local entrypoint (fixed, see above)
```

## Launching training

### Single-GPU (the fixed path — use this for retina training)

```bash
python main.py \
  --fname configs/your_config.yaml \
  --devices cuda:0
```

### Multi-GPU (unmodified upstream, local)

```bash
python main.py \
  --fname configs/your_config.yaml \
  --devices cuda:0 cuda:1 cuda:2
```

### Multi-GPU / SLURM (unmodified upstream)

```bash
python main_distributed.py \
  --fname configs/your_config.yaml \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
```

Config files hold all experiment parameters (no CLI hyperparameter flags) — see `configs/` for
examples.

## Requirements

* Python 3.8+
* PyTorch 2.0+, torchvision
* pyyaml, numpy, opencv, submitit (submitit only needed for `main_distributed.py`)

## Cloning

This repo is **public** — plain anonymous clone works everywhere, no credential setup needed:

```bash
git clone https://github.com/khalilLaatiris/ijepa-core.git /home/$USER/retina/repos/ijepa-core
```

Later updates: `cd /home/$USER/retina/repos/ijepa-core && git pull`.

## Citation

Original method — please cite the upstream paper if you use this code:

```
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```

## License

Attribution-NonCommercial 4.0 International — see [LICENSE](./LICENSE) (same as upstream). Research
use only, no commercial use.
