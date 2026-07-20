"""Hyperparameter-fingerprinted cache for trained downstream heads.

Ported from Code/cswin_jepa_ret_pipeline_final_from_literature.ipynb (cell 3c). Adapted: the
source notebook's functions read module-level Colab-cell globals (MODELS_ROOT, CACHE_POLICY);
here both are constructor arguments on a small class instead, so this works identically in a
plain script, a pytest run, or a Colab cell that passes its own Drive path -- same caching
behavior, explicit interface instead of hidden global state.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path


class HeadCache:
    def __init__(self, models_root, policy="skip"):
        """policy: 'ask' (interactive prompt on a hit), 'skip' (reuse silently), 'retrain' (always retrain)."""
        assert policy in ("ask", "skip", "retrain"), f"unknown policy: {policy}"
        self.models_root = Path(models_root)
        self.policy = policy

    def ckpt_stamp(self, ckpt_path):
        """Encoder-checkpoint identity, folded into every fingerprint. Without it, retraining an
        encoder to the SAME path fingerprints identically and the cache serves metrics computed
        with the OLD encoder. stat() only -- no read."""
        p = Path(ckpt_path)
        if not p.exists():
            return None
        st = p.stat()
        return dict(path=str(p), size=st.st_size, mtime=int(st.st_mtime))

    def _fp(self, cfg):
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _dir(self, task, enc, head, seed):
        d = self.models_root / task / enc / head / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def check(self, task, enc, head, seed, cfg):
        d = self._dir(task, enc, head, seed)
        mp = d / "meta.json"
        tag = f"{task}/{enc}/{head}/seed{seed}"
        if not mp.exists():
            return True, None
        meta = json.loads(mp.read_text())
        if meta.get("fingerprint") != self._fp(cfg):
            print(f"  [{tag}] hyperparams/encoder changed -- retraining.")
            return True, None
        if self.policy == "retrain":
            return True, None
        if self.policy == "skip":
            print(f"  [{tag}] reusing cache ({meta.get('saved_at', '?')}).")
            return False, meta
        if input(f"  [{tag}] identical cached head ({meta.get('saved_at', '?')}). Rerun? [y/N] ").strip().lower() == "y":
            return True, None
        print(f"  [{tag}] skipped.")
        return False, meta

    def save(self, task, enc, head, seed, cfg, metrics, save_fn, arch=None):
        """`arch` = the rebuild recipe. A bare state_dict is NOT enough for cold inference:
        FPNDecoder's in_chs are only discoverable by a dummy forward through the encoder."""
        d = self._dir(task, enc, head, seed)
        save_fn(d)
        (d / "meta.json").write_text(json.dumps(
            dict(fingerprint=self._fp(cfg), saved_at=datetime.now().isoformat(timespec="seconds"),
                 cfg=cfg, arch=arch, metrics=metrics), indent=2, default=str))
