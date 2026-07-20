"""I-JEPA pretraining — single GPU, original (vendored) ImageNet-style dataset loader.

Local, self-contained recreation of facebookresearch/ijepa's main.py/src/train.py loop
(same masking/EMA/loss/LR-WD-schedule math, published default hyperparameters, same
dataset loader) — does not clone or execute the original repo. See README.md for what's
vendored vs adapted and why.

Usage:
    python main.py --fname config.yaml --device cuda:0
"""

import argparse
import pprint

import torch
import yaml

from src.train import main as train_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str, default='config.yaml',
    help='path to the training config yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help="device to train on, e.g. 'cuda:0' or 'cpu'")


def main():
    args = parser.parse_args()

    with open(args.fname, 'r') as f:
        params = yaml.safe_load(f)

    print(f'called-params {args.fname}')
    pprint.PrettyPrinter(indent=4).pprint(params)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'--device {args.device} requested but torch.cuda.is_available() is False')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    train_main(args=params, device=device)


if __name__ == '__main__':
    main()
