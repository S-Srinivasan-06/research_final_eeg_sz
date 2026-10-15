"""Checkpoints that let any run stop and continue exactly.

`last.pt` holds model, optimiser, scheduler, epoch, step-within-epoch, early-stopping state, RNG states and any
extra dict; it is written atomically (tmp file + rename) at the end of every epoch and every `every_steps` steps.
`progress.json` mirrors the small fields (epoch, step, best metric, wall time) in plain text for quick inspection.
`best.pt` is the model at the best selection metric.
Mid-epoch resume: the batch order of epoch e depends only on (seed, e) (linoss/data.py), so the trainer
skips the batches already done (FromBatch in linoss/train.py). last.pt is loaded with weights_only=False because it
holds Python RNG states; it is only ever read back by the run that wrote it.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, path)


class Checkpointer:
    def __init__(self, out_dir, log, every_steps=0):
        self.out = Path(out_dir); self.log = log; self.every_steps = every_steps
        self.last, self.best = self.out / "last.pt", self.out / "best.pt"

    def save(self, *, model, optimizer, scheduler, epoch, step, best, extra=None, why="epoch"):
        t = time.time()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "epoch": epoch, "step": step, "best": best, "rng": rng_state(), "extra": extra or {}}
        atomic_save(state, self.last)
        prog = {"epoch": epoch, "step": step, "best": best, "why": why, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.out / "progress.json").write_text(json.dumps(prog, indent=2, default=float))
        self.log.event("checkpoint", why=why, epoch=epoch, step=step, sec=round(time.time() - t, 2),
                       mb=round(self.last.stat().st_size / 1e6, 1))

    def save_best(self, model, epoch, metric):
        atomic_save({"model": model.state_dict(), "epoch": epoch, "metric": metric}, self.best)
        self.log.event("best_model", epoch=epoch, metric=metric)

    def resume(self, model, optimizer=None, scheduler=None, src=None):
        p = Path(src) / "last.pt" if src else self.last
        if not p.exists():
            self.log("no checkpoint to resume from; starting fresh")
            return None
        ck = torch.load(p, map_location=next(model.parameters()).device, weights_only=False)
        model.load_state_dict(ck["model"])
        if optimizer is not None:
            optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and ck["scheduler"] is not None:
            scheduler.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"])
        self.log.event("resumed", src=str(p), epoch=ck["epoch"], step=ck["step"], best=ck["best"])
        return ck


