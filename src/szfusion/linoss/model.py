"""OscSeq: the LinOSS branch of the CatBoost × LinOSS fusion. It outputs one seizure logit per second.

Data flow (shapes for one batch; B segments of L seconds, N electrodes, F = 100 spectral bins):

  x (B, L, N, 100)            per-second tokens (tokens.py): z-scored log|FFT| of each electrode, 1 Hz bins 0-99 Hz
  -> Linear(100 -> 32) + electrode embedding (N, 32)          spectral front end, shared over electrodes
  -> n1 = 1 BiLinOSS block per electrode   (B*N, L, 32)       P = 32 oscillators per direction, shared weights
  -> [mean ; max] over electrodes -> Linear(64 -> 96)         spatial pooling (does not assume a seizure location)
  -> n2 = 2 BiLinOSS blocks                (B, L, 96)         P = 96 oscillators per direction
  -> Linear(96 -> 1)                       (B, L)             per-second seizure logit

A BiLinOSS block is the official LinOSS block (Rusch & Rus, ICLR 2025): BatchNorm without affine parameters, then
the SSM, GELU, dropout, GLU, dropout and a residual connection. Here the SSM is a forward LinOSS layer plus a
time-reversed LinOSS layer with its own oscillators. Detection is offline, so each second sees the future part of
its window as well as the past, like the CatBoost branch's centred ±7-window context.

At 1 token per second the oscillators act on the slow evolution of the spectra over seconds to minutes (build-up,
evolution, post-ictal slowing). The EEG rhythms themselves are already resolved by the FFT bins. The final recipe
initialises the natural periods log-uniformly in 5-240 s (osc_periods=(5, 240)).

Parameter names are kept identical to the development code, so its checkpoints load unchanged (path.md S1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from szfusion.linoss.layer import GLU, LinOSSLayer


class BiLinOSSBlock(nn.Module):
    """BatchNorm (no affine) -> forward LinOSS + time-reversed LinOSS -> GELU -> dropout -> GLU -> dropout -> residual.
    Padded steps (mask = 0) are zeroed before the SSM, so padding never feeds the oscillators."""

    def __init__(self, H, P, disc="IM", drop=0.1, periods=None):
        super().__init__()
        self.norm = nn.BatchNorm1d(H, affine=False)
        self.fwd = LinOSSLayer(P, H, disc, periods)
        self.bwd = LinOSSLayer(P, H, disc, periods)
        self.glu = GLU(H)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):                                              # x (N, L, H), mask (N, L)
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            h = h * mask[..., None]
        s = self.fwd(h) + self.bwd(h.flip(1)).flip(1)
        h = self.drop(F.gelu(s))
        return x + self.drop(self.glu(h))


class OscSeq(nn.Module):
    def __init__(self, n_bins=100, n_el=19, e_dim=32, p1=32, n1=1, H=96, p2=96, n2=2, disc="IM", drop=0.1,
                 osc_periods=None):
        """n_el: number of input channels (19 referential electrodes on TUSZ; the bipolar channels on CHB-MIT).
        osc_periods=None gives the official LinOSS initialisation; (t_min, t_max) gives log-uniform natural periods
        in seconds."""
        super().__init__()
        per = tuple(osc_periods) if osc_periods else None
        self.front = nn.Linear(n_bins, e_dim)
        self.el_emb = nn.Parameter(torch.zeros(n_el, e_dim))
        self.el_blocks = nn.ModuleList([BiLinOSSBlock(e_dim, p1, disc, drop, per) for _ in range(n1)])
        self.pool = nn.Linear(2 * e_dim, H)
        self.seq_blocks = nn.ModuleList([BiLinOSSBlock(H, p2, disc, drop, per) for _ in range(n2)])
        self.head = nn.Linear(H, 1)

    def forward(self, x, mask=None):
        """x (B, L, N, 100), mask (B, L) with 1 = a real second -> logits (B, L)."""
        return self.head(self.features(x, mask)).squeeze(-1)

    def features(self, x, mask=None):
        """The per-second summary that the head reads: (B, L, H). The 12 s / 60 s overlookers read it too."""
        B, L, N, Fd = x.shape
        h = self.front(x) + self.el_emb                                           # (B, L, N, E)
        h = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        m_el = None if mask is None else mask[:, None, :].expand(B, N, L).reshape(B * N, L)
        for blk in self.el_blocks:
            h = blk(h, m_el)
        h = h.view(B, N, L, -1)
        h = self.pool(torch.cat([h.mean(1), h.amax(1)], -1))                       # (B, L, H)
        for blk in self.seq_blocks:
            h = blk(h, mask)
        return h


def build(cfg=None):
    return OscSeq(**(cfg or {}))


@torch.no_grad()
def oscillator_report(model, fs=1.0):
    """Natural frequency of every oscillator at the token rate (1 Hz): quantiles in mHz, median period in s."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, LinOSSLayer):
            f = m.frequencies_hz(fs).cpu().numpy()
            q = np.quantile(f, [0, .1, .5, .9, 1])
            out[name] = {"mHz_q0_10_50_90_100": (1000 * q).round(1).tolist(),
                         "median_period_s": round(float(1 / max(np.median(f), 1e-9)), 1)}
    return out
