# CLAUDE.md

## What this is

Canonical single-GPU I-JEPA pretraining engine (ViT only) for the retina-diseases research thread.
Umbrella-root subproject — see `../CLAUDE.md` for how it relates to `Code/`, `+ shaped JEPA/`, etc.

## Freeze rule — read before editing `pretrain/`

`pretrain/` (everything under it: `main.py`, `config.yaml`, `src/**`) is frozen once BOTH test-gate
stages pass:

1. `tests/smoke_test.py` (CPU, synthetic data) — passing as of the commit that introduced it.
2. `colab/confirm_train.ipynb` (GPU, real stratified-subset data, 5 epochs, ViT-S) — run manually by
   Khalil on Colab; check this file's own status note / commit history for pass confirmation before
   assuming it has run.

Once both have passed: **do not silently edit anything under `pretrain/`.** No refactors, no
"cleanup," no dependency bumps, no touching vendored files even for style. A bugfix or behavior
change requires an explicit user request naming the exact file and change. New capability (new
model size support, new config knob) goes in new files or clearly-scoped additive changes — never
a rewrite of tested core.

`downstream/` is NOT covered by this rule — it's expected to grow (new tasks, new decoders,
structured for exactly that).

## Status

- Smoke test: PASSED (commit f26e4d9, `tests/smoke_test.py`). CPU, synthetic data, 2 epochs
  (1 fresh + 1 resumed). Run 1 loss: 0.2698. Run 2 (resumed) loss: 0.1814. Checkpoint
  save/resume verified — all keys matched on reload.
- Colab confirm run (`colab/confirm_train.ipynb`, GPU, real data, 5 epochs, ViT-S): not yet
  executed — this file does not exist yet (a later task/plan adds it). `pretrain/` is NOT
  frozen until this second gate also passes and this file is updated to say so.
