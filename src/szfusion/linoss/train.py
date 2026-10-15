"""Train one seed of OscSeq (the LinOSS branch) on the per-second token store; write its dev and eval tracks.

  python -m szfusion.linoss.train --tokens <work>/tokens/tusz --out <work>/tusz/linoss/s0 --seed 0
  (the same command resumes an interrupted run exactly; --smoke runs a tiny end-to-end check)

Defaults = the final recipe (path.md 4.2):
  * Every epoch draws 6,000 fresh segments of 240 s (data.TrainSegments).
  * Loss: per-second BCE over real seconds.
  * Optimiser: AdamW, lr 3e-4, linear warm-up over the first 300 steps, then cosine decay over 30 epochs.
  * Weight decay 0.01 on weight matrices and 0 on the LinOSS input/output maps B and C. No decay on the oscillator
    parameters, biases or electrode embedding.
  * Gradient clipping at 5.
  * EMA of the weights with decay 0.998. The EMA weights are the ones evaluated and saved.
  * Early stopping on dev per-second AUPRC with patience 6.
  * Natural periods initialised log-uniform in 5-240 s ({"osc_periods": [5, 240]}).
Dev is used only to pick the epoch. Eval probabilities are written with the best-on-dev weights, but eval labels are
never read here: they are scored once, by szfusion.fusion.
Outputs in --out:
  best.pt, last.pt                 weights; the resumable state
  prob_dev.npy, prob_eval.npy      per-second probability aligned to the store rows (NaN = recording not used),
                                   from 240 s windows every 120 s, blended (data.RecordingWindows)
  results.json                     configuration, best epoch, dev per-second AUROC/AUPRC, learned oscillator periods
  run.log, events.jsonl            full log (contains paths: never committed)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from szfusion.linoss.data import RecordingWindows, Store, TrainSegments
from szfusion.linoss.model import build, oscillator_report
from szfusion.util.ckpt import Checkpointer
from szfusion.util.monitor import Log, ModuleTimer, Profiler, resources


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="token store root (contains train/ dev/ eval/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=240, help="segment / window length in seconds")
    ap.add_argument("--n_seg", type=int, default=6000, help="fresh training segments per epoch")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--wd_ssm", type=float, default=0.0, help="weight decay of the LinOSS B and C")
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.998, help="EMA decay of the evaluated weights (0 = off)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--model", default='{"osc_periods": [5, 240]}', help="JSON keyword arguments of OscSeq")
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--final_stride", type=int, default=120, help="window stride (s) of the final dev/eval tracks")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=50, help="optimiser steps between checkpoints (0 = epoch only)")
    ap.add_argument("--max_hours", type=float, default=None, help="stop cleanly (resumable) after this many hours")
    ap.add_argument("--cpu", action="store_true", help="stay on CPU even if a GPU is visible")
    ap.add_argument("--deterministic", action="store_true", help="request deterministic CUDA kernels (slower)")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args(argv)


class FromBatch(Sampler):
    """Sequential order over the (already shuffled) epoch spec, starting at a given batch (exact mid-epoch resume)."""

    def __init__(self, n, bs):
        self.n, self.bs, self.start = n, bs, 0

    def __iter__(self):
        return iter(range(self.start * self.bs, self.n))

    def __len__(self):
        return max(self.n - self.start * self.bs, 0)


def param_groups(model, wd, wd_ssm):
    """Weight decay wd on weight matrices, wd_ssm on the LinOSS B and C, none on the oscillator parameters (A, steps),
    D, biases and the electrode embedding."""
    decay, ssm, no = [], [], []
    for n, p in model.named_parameters():
        if n.endswith(".B") or n.endswith(".C"):
            ssm.append(p)
        elif p.ndim >= 2 and "el_emb" not in n:
            decay.append(p)
        else:
            no.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": ssm, "weight_decay": wd_ssm},
            {"params": no, "weight_decay": 0.0}]


class EMA:
    """Exponential moving average of the parameters (buffers such as BatchNorm statistics are copied)."""

    def __init__(self, model, decay):
        import copy
        self.decay, self.model = decay, copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


@torch.no_grad()
def predict(model, ds, bs, workers, prof, log, tag):
    model.eval()
    out = np.zeros((len(ds), ds.L), np.float32)
    t_wait = time.perf_counter(); t0 = time.time()
    dl = DataLoader(ds, bs, shuffle=False, num_workers=workers, persistent_workers=False,
                    generator=torch.Generator())     # own generator: creating a loader must not draw from the global
                                                     # torch RNG (that would shift every later dropout mask on resume)
    dev = next(model.parameters()).device
    for bi, (x, m, idx) in enumerate(dl):
        prof.add(f"{tag}_data", time.perf_counter() - t_wait)
        with prof.stage(f"{tag}_forward"):
            out[idx.numpy()] = model(x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)).float().cpu().numpy()
        if bi % 50 == 0:
            log(f"  [{tag}] {min((bi + 1) * bs, len(ds))}/{len(ds)} windows  {time.time() - t0:.0f}s  {resources()}")
        t_wait = time.perf_counter()
    return out


def sec_metrics(prob, y):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = ~np.isnan(prob)
    return {"auroc": float(roc_auc_score(y[ok], prob[ok])), "auprc": float(average_precision_score(y[ok], prob[ok])),
            "seconds": int(ok.sum())}


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = Log(out); prof = Profiler(log)
    log.event("config", **vars(a), torch=torch.__version__)
    if (out / "results.json").exists():
        log("results.json exists: run finished. Delete it to re-score."); log.close(); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    mcfg = json.loads(a.model)
    with prof.stage("load_data"):
        st_tr, st_dev, st_ev = (Store(s, a.tokens) for s in ("train", "dev", "eval"))
        n_seg = 64 if a.smoke else a.n_seg
        tr = TrainSegments(st_tr, a.L, n_seg, a.seed)
        sub = list(range(0, len(st_dev.index), 40)) if a.smoke else None
        dev_win = RecordingWindows(st_dev, a.L, a.L, sub)                     # per-epoch selection: stride = L
        y_dev = np.asarray(st_dev.Y, np.int8)
    log.event("data", train_recordings=len(st_tr.index), train_seizure_seconds=tr.n_pos_sec, segments_per_epoch=n_seg,
              segment_s=a.L, dev_recordings=len(st_dev.index), dev_windows_per_epoch=len(dev_win),
              eval_recordings=len(st_ev.index))
    epochs = 2 if a.smoke else a.epochs

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model = build({"n_el": tr.shape[0], **mcfg}).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(param_groups(model, a.wd, a.wd_ssm), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ck = Checkpointer(out, log, a.ckpt_every)
    log.event("model", name="OscSeq", params=n_params, cfg=mcfg, threads=a.threads, device=str(device),
              gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    log.event("oscillators_init", **oscillator_report(model))
    state = ck.resume(model, opt, sched)
    ema = EMA(model, a.ema) if a.ema > 0 else None
    if ema and state and "ema" in state["extra"]:
        ema.model.load_state_dict(state["extra"]["ema"])
    xtra = lambda done: {"epoch_complete": done, **({"ema": ema.model.state_dict()} if ema else {})}  # noqa: E731
    best = {"metric": -1.0, "epoch": 0, "patience": 0}
    epoch, start_batch = 1, 0
    if state:
        best = state["best"]
        epoch, start_batch = (state["epoch"] + 1, 0) if state["extra"].get("epoch_complete") else (state["epoch"], state["step"])
    sampler = FromBatch(len(tr), a.bs)
    mt = ModuleTimer(model, ["front", "el_blocks", "pool", "seq_blocks", "head"], log)
    t_run = time.time()
    n_batches = (len(tr) + a.bs - 1) // a.bs
    while epoch <= epochs and best["patience"] < a.patience:
        tr.set_epoch(epoch); sampler.start = start_batch
        loader = DataLoader(tr, a.bs, sampler=sampler, num_workers=a.num_workers, persistent_workers=False,
                            pin_memory=device.type == "cuda", generator=torch.Generator())
        model.train(); prof.reset()
        n_seen, run_l, nb, t_log = 0, 0.0, 0, time.time()
        log(f"epoch {epoch}: {n_batches} batches of {a.bs} x {a.L} s, starting at batch {start_batch}, "
            f"lr {opt.param_groups[0]['lr']:.2e}")
        t_wait = time.perf_counter()
        for bi, (x, y, m) in enumerate(loader, start=start_batch):
            prof.add("data_wait", time.perf_counter() - t_wait)
            x, y, m = (t.to(device, non_blocking=True) for t in (x, y, m))
            with prof.stage("forward+loss"):
                logit = model(x, m)
                loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum().clamp(min=1)
            with prof.stage("backward"):
                opt.zero_grad(set_to_none=True); loss.backward()
            with prof.stage("optim"):
                gn = float(nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip))
                g_step = (epoch - 1) * n_batches + bi
                warm = min(1.0, (g_step + 1) / a.warmup_steps) if a.warmup_steps else 1.0
                base = [g["lr"] for g in opt.param_groups]
                for g in opt.param_groups:
                    g["lr"] *= warm
                opt.step()
                for g, b in zip(opt.param_groups, base):                  # keep the cosine schedule's own lr
                    g["lr"] = b
                if ema:
                    ema.update(model)
            lv = loss.item()
            if not np.isfinite(lv):
                log.event("nonfinite_loss", epoch=epoch, batch=bi); raise SystemExit(1)
            n_seen += len(x); run_l += lv; nb += 1
            step = bi + 1; last = step == n_batches
            if step % a.log_every == 0 or last:
                log(f"  ep {epoch} step {step}/{n_batches}  loss {run_l / max(nb, 1):.4f}  grad_norm {gn:.2f}  "
                    f"{n_seen * a.L / (time.time() - t_log):.0f} EEG-s/s  {resources()}")
                run_l, nb = 0.0, 0
            if a.ckpt_every and step % a.ckpt_every == 0 and not last:
                with prof.stage("checkpoint"):
                    ck.save(model=model, optimizer=opt, scheduler=sched, epoch=epoch, step=step, best=best,
                            extra=xtra(False), why="steps")
            if a.max_hours and (time.time() - t_run) / 3600 > a.max_hours and not last:
                ck.save(model=model, optimizer=opt, scheduler=sched, epoch=epoch, step=step, best=best,
                        extra=xtra(False), why="max_hours")
                log.event("stopped_for_time", epoch=epoch, batch=step, note="re-run the same command to resume")
                prof.report(f"epoch {epoch} (partial)", n_seen); mt.report(f"epoch {epoch}"); log.close()
                return
            t_wait = time.perf_counter()
        start_batch = 0
        prof.report(f"epoch {epoch} train", n_seen)
        mt.report(f"epoch {epoch} train")

        prof.reset()
        sel_model = ema.model if ema else model
        lg = predict(sel_model, dev_win, a.eval_bs, a.num_workers, prof, log, "dev")
        md = sec_metrics(dev_win.blend(lg, len(y_dev)), y_dev)
        if md["auprc"] > best["metric"]:
            best.update(metric=md["auprc"], epoch=epoch, patience=0)
            ck.save_best(sel_model, epoch, md["auprc"])
        else:
            best["patience"] += 1
        log.event("epoch_done", epoch=epoch, dev_auprc=round(md["auprc"], 4), dev_auroc=round(md["auroc"], 4),
                  best_epoch=best["epoch"], best=round(best["metric"], 4), patience=best["patience"],
                  lr=opt.param_groups[0]["lr"], hours=round((time.time() - t_run) / 3600, 2), **resources())
        log.event("oscillators", epoch=epoch, **oscillator_report(model))
        prof.report(f"epoch {epoch} dev eval")
        sched.step()
        ck.save(model=model, optimizer=opt, scheduler=sched, epoch=epoch, step=0, best=best,
                extra=xtra(True), why="epoch")
        epoch += 1
    mt.remove()

    log(f"training finished: best epoch {best['epoch']} dev per-second AUPRC {best['metric']:.4f}")
    model.load_state_dict(torch.load(ck.best, map_location=device, weights_only=True)["model"])
    res = {"seed": a.seed, "params": n_params, "best": best, "cfg": {"n_el": tr.shape[0], **mcfg}, "L": a.L,
           "final_stride": a.final_stride}
    prof.reset()
    for split, st in (("dev", st_dev), ("eval", st_ev)):
        f = out / f"prob_{split}.npy"
        if not f.exists():
            subset = sub if split == "dev" else (list(range(0, len(st.index), 40)) if a.smoke else None)
            ds = RecordingWindows(st, a.L, a.final_stride, subset)
            p = ds.blend(predict(model, ds, a.eval_bs, a.num_workers, prof, log, f"final_{split}"), len(st.Y))
            np.save(f, p.astype(np.float32))
        if split == "dev":                      # eval labels are NOT read here (eval is scored once, by fusion)
            res[split] = sec_metrics(np.load(f).astype(np.float64), np.asarray(st.Y, np.int8))
            log.event("final_scores", split=split, **res[split])
        else:
            log.event("saved_eval_probs", note="not scored here")
    res["oscillators"] = oscillator_report(model)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    log.event("done", file="results.json", hours=round((time.time() - t_run) / 3600, 2))
    prof.report("final scoring")
    log.close()


if __name__ == "__main__":
    main()
