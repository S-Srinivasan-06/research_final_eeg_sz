"""Verbose logging and bottleneck finding.

Log        prints every line with elapsed time, mirrors it to <out>/run.log, and writes structured events to
           <out>/events.jsonl (one JSON object per event) so a run can be inspected or plotted afterwards.
Profiler   wall-clock time per pipeline stage (data_wait, to_device, forward, backward, optim, eval, ckpt, ...).
           `with prof.stage("forward"):` ... ; `prof.report("epoch 3")` prints a table sorted by time with each
           stage's share of the wall time and names the largest one as the bottleneck. With sync=True the CUDA
           queue is flushed at every stage boundary so GPU work is charged to the stage that launched it
           (accurate, slightly slower; use it for profiling runs).
ModuleTimer forward time per named submodule (e.g. backbone, hyper.0, hyper.1) via CUDA events or perf_counter.
resources  GPU memory (current / peak), process RSS, throughput.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch


class Log:
    def __init__(self, out_dir, name="run", echo=True):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.echo = echo
        self._txt = open(self.out / f"{name}.log", "a", encoding="utf-8", buffering=1)
        self._ev = open(self.out / "events.jsonl", "a", encoding="utf-8", buffering=1)
        self("=" * 100)
        self(f"start {time.strftime('%Y-%m-%d %H:%M:%S')}  pid={os.getpid()}  python={sys.version.split()[0]}  "
             f"torch={torch.__version__}  cuda={torch.cuda.is_available()}"
             + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    def __call__(self, msg=""):
        el = time.time() - self.t0
        line = f"[{el:8.1f}s] {msg}"
        if self.echo:
            print(line, flush=True)
        self._txt.write(line + "\n")

    def event(self, kind, **fields):
        """Structured record (also printed)."""
        rec = {"t": round(time.time() - self.t0, 2), "kind": kind, **fields}
        self._ev.write(json.dumps(rec, default=_jsonable) + "\n")
        self(f"{kind}: " + " ".join(f"{k}={_fmt(v)}" for k, v in fields.items()))

    def table(self, title, header, rows):
        widths = [max(len(str(h)), *(len(_fmt(r[i])) for r in rows)) if rows else len(str(h))
                  for i, h in enumerate(header)]
        self(f"--- {title}")
        self("  " + "  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
        for r in rows:
            self("  " + "  ".join(_fmt(v).ljust(w) for v, w in zip(r, widths)))

    def close(self):
        self._txt.close(); self._ev.close()


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}"
    return str(v)


def _jsonable(o):
    try:
        return float(o)
    except Exception:  # noqa: BLE001
        return str(o)


class Profiler:
    def __init__(self, log: Log, sync: bool = False):
        self.log, self.sync = log, sync and torch.cuda.is_available()
        self.reset()

    def reset(self):
        self.tot = defaultdict(float); self.cnt = defaultdict(int); self.t_start = time.perf_counter()

    @contextmanager
    def stage(self, name):
        if self.sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            self.tot[name] += time.perf_counter() - t
            self.cnt[name] += 1

    def add(self, name, seconds):
        self.tot[name] += seconds; self.cnt[name] += 1

    def report(self, title, n_samples=None):
        wall = time.perf_counter() - self.t_start
        covered = sum(self.tot.values())
        rows = []
        for k, v in sorted(self.tot.items(), key=lambda kv: -kv[1]):
            rows.append([k, round(v, 2), f"{100 * v / max(wall, 1e-9):.1f}%", round(1000 * v / max(self.cnt[k], 1), 2),
                         self.cnt[k]])
        rows.append(["(untracked)", round(wall - covered, 2), f"{100 * (wall - covered) / max(wall, 1e-9):.1f}%", "", ""])
        self.log.table(f"time by stage, {title} (wall {wall:.1f}s"
                       + (f", {n_samples / wall:.0f} samples/s" if n_samples else "") + ")",
                       ["stage", "sec", "share", "ms/call", "calls"], rows)
        if self.tot:
            top = max(self.tot.items(), key=lambda kv: kv[1])
            self.log.event("bottleneck", scope=title, stage=top[0], share=round(top[1] / max(wall, 1e-9), 3),
                           wall_sec=round(wall, 1), sync=self.sync)
        return dict(self.tot), wall


class ModuleTimer:
    """Forward time per named submodule. Uses CUDA events on GPU (no sync per call), perf_counter on CPU."""

    def __init__(self, model, names, log: Log):
        self.log, self.names = log, names
        self.cuda = next(model.parameters()).is_cuda
        self.tot = defaultdict(float); self.cnt = defaultdict(int); self._pending = []
        mods = dict(model.named_modules())
        self.handles = []
        for n in names:
            if n not in mods:
                log(f"[ModuleTimer] no submodule named {n!r}; available: {[k for k in mods if k][:30]}")
                continue
            self.handles.append(mods[n].register_forward_pre_hook(self._pre(n)))
            self.handles.append(mods[n].register_forward_hook(self._post(n)))
        self._start = {}

    def _pre(self, n):
        def f(_m, _i):
            if self.cuda:
                e = torch.cuda.Event(enable_timing=True); e.record(); self._start[n] = e
            else:
                self._start[n] = time.perf_counter()
        return f

    def _post(self, n):
        def f(_m, _i, _o):
            if self.cuda:
                e = torch.cuda.Event(enable_timing=True); e.record(); self._pending.append((n, self._start[n], e))
            else:
                self.tot[n] += time.perf_counter() - self._start[n]; self.cnt[n] += 1
        return f

    def flush(self):
        if self._pending:
            torch.cuda.synchronize()
            for n, a, b in self._pending:
                self.tot[n] += a.elapsed_time(b) / 1000.0; self.cnt[n] += 1
            self._pending.clear()

    def report(self, title):
        self.flush()
        total = sum(self.tot.values())
        rows = [[n, round(self.tot[n], 3), f"{100 * self.tot[n] / max(total, 1e-9):.1f}%",
                 round(1000 * self.tot[n] / max(self.cnt[n], 1), 3)] for n in self.names if self.cnt[n]]
        self.log.table(f"forward time by module, {title}", ["module", "sec", "share of timed", "ms/call"], rows)
        self.tot.clear(); self.cnt.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


def resources():
    r = {}
    if torch.cuda.is_available():
        r["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
        r["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    try:
        import psutil
        r["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 3)
    except ImportError:
        pass
    return r
