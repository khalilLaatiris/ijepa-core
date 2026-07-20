"""Tests for downstream/cache.py — HeadCache round-trip and fingerprint invalidation."""
import time

from downstream.cache import HeadCache


def test_cache_miss_then_hit(tmp_path):
    cache = HeadCache(tmp_path, policy="skip")
    cfg = dict(lr=1e-3, epochs=10)

    train, cached = cache.check("taskA", "vit_small", "linear", 0, cfg)
    assert train is True
    assert cached is None

    saved_files = []
    cache.save("taskA", "vit_small", "linear", 0, cfg, metrics={"acc": 0.9},
               save_fn=lambda d: saved_files.append((d / "weights.pth").write_text("dummy")))
    assert saved_files  # save_fn was actually invoked

    train, cached = cache.check("taskA", "vit_small", "linear", 0, cfg)
    assert train is False
    assert cached["metrics"] == {"acc": 0.9}


def test_cache_invalidates_on_changed_cfg(tmp_path):
    cache = HeadCache(tmp_path, policy="skip")
    cfg1 = dict(lr=1e-3)
    cfg2 = dict(lr=5e-4)  # different hyperparameter -> different fingerprint

    cache.save("taskA", "vit_small", "linear", 0, cfg1, metrics={"acc": 0.9}, save_fn=lambda d: None)
    train, cached = cache.check("taskA", "vit_small", "linear", 0, cfg2)
    assert train is True
    assert cached is None


def test_cache_policy_retrain_always_forces_retrain(tmp_path):
    cache = HeadCache(tmp_path, policy="retrain")
    cfg = dict(lr=1e-3)
    cache.save("taskA", "vit_small", "linear", 0, cfg, metrics={"acc": 0.9}, save_fn=lambda d: None)

    retrain_cache = HeadCache(tmp_path, policy="retrain")
    train, cached = retrain_cache.check("taskA", "vit_small", "linear", 0, cfg)
    assert train is True


def test_ckpt_stamp_changes_with_file_mtime(tmp_path):
    cache = HeadCache(tmp_path)
    ckpt = tmp_path / "encoder.pth"
    ckpt.write_text("v1")
    stamp1 = cache.ckpt_stamp(ckpt)

    time.sleep(1.1)  # mtime resolution on some filesystems is ~1s
    ckpt.write_text("v2 -- different size too")
    stamp2 = cache.ckpt_stamp(ckpt)

    assert stamp1 != stamp2
    assert cache.ckpt_stamp(tmp_path / "missing.pth") is None
