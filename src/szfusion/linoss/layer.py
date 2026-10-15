"""LinOSS (Rusch & Rus, ICLR 2025, "Oscillatory State-Space Models") in PyTorch.

Port of the official JAX code (github.com/tk-rusch/linoss, models/LinOSS.py): same parameters, initialisation,
discretisations and block. Each state dimension is a forced harmonic oscillator

    y''(t) = -A y(t) + B u(t),    x(t) = Re(C y(t)) + D u(t),    A = relu(A_raw) >= 0 (diagonal),  B, C complex

discretised with step dt = sigmoid(steps_raw) per oscillator, either implicitly (IM: always stable, damped,
|lambda|^2 = 1 / (1 + dt^2 A)) or implicit-explicit (IMEX: energy-conserving while dt^2 A <= 4). In the official
code the 2x2 recurrence per oscillator runs as an associative scan. Here the same recurrence is computed as a causal
convolution, which is exact because LinOSS is time-invariant (unlike Mamba, nothing depends on the input):
  kernel h[k] = (M^k f)_y      (M^k by repeated squaring, no eigen-decomposition, stable for A = 0)
  y = h * (B u)                (FFT for long sequences, direct for short ones)
`linoss_reference` runs the recurrence step by step; tests/test_linoss.py checks that every path agrees with it.

Oscillator frequency per step: IM theta = atan(dt sqrt(A)); at 200 Hz input that is f = 200 theta / (2 pi) Hz.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _transition(A, dt, disc):
    """Per-oscillator 2x2 matrix M (P, 2, 2) on state (z, y) and input vector f (P, 2)."""
    if disc == "IM":
        S = 1.0 / (1.0 + dt ** 2 * A)
        M = torch.stack([torch.stack([S, -dt * A * S], -1), torch.stack([dt * S, S], -1)], -2)
        f = torch.stack([dt * S, dt ** 2 * S], -1)
    elif disc == "IMEX":
        one = torch.ones_like(A)
        M = torch.stack([torch.stack([one, -dt * A], -1), torch.stack([dt, 1.0 - dt ** 2 * A], -1)], -2)
        f = torch.stack([dt, dt ** 2], -1)
    else:
        raise ValueError(disc)
    return M, f


def linoss_kernel(A, dt, L, disc="IM"):
    """(P, L) impulse response of the position y to (B u): h[k] = (M^k f)[y]."""
    M, f = _transition(A, dt, disc)
    V, Mp = f[:, None, :], M                                     # V[:, k] = M^k f
    while V.shape[1] < L:
        V = torch.cat([V, torch.einsum("pij,pkj->pki", Mp, V)], 1)
        Mp = Mp @ Mp
    return V[:, :L, 1]


def causal_conv(x, h):
    """x (..., P, L), h (P, L) -> y[..., l] = sum_{k<=l} h[k] x[l-k]."""
    L = x.shape[-1]
    if L <= 64:
        idx = torch.arange(L, device=x.device)
        d = idx[:, None] - idx[None, :]                                          # l - j
        T = torch.where(d >= 0, h[:, d.clamp(min=0)], torch.zeros((), device=x.device, dtype=h.dtype))  # (P, L, L)
        return torch.einsum("...pj,plj->...pl", x, T)
    n = 1 << (2 * L - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(h, n=n), n=n)
    return y[..., :L]


class LinOSSLayer(nn.Module):
    def __init__(self, ssm_size, H, discretization="IM", periods=None):
        """periods=None: the official initialisation (A ~ U(0, 1), steps ~ U(0, 1)). periods=(t_min, t_max): natural
        periods (in steps) log-uniform in [t_min, t_max]; A is solved from the period and the sampled step, so only A
        changes. IM needs t_min > 4 (theta = atan(dt sqrt(A)) < pi/2), IMEX t_min > 2."""
        super().__init__()
        P = ssm_size
        self.disc = discretization
        self.A_diag = nn.Parameter(torch.rand(P))                                       # uniform(0, 1)
        self.B = nn.Parameter((torch.rand(P, H, 2) * 2 - 1) / math.sqrt(H))           # simple_uniform_init
        self.C = nn.Parameter((torch.rand(H, P, 2) * 2 - 1) / math.sqrt(P))
        self.D = nn.Parameter(torch.randn(H))
        self.steps = nn.Parameter(torch.rand(P))
        if periods is not None:
            t_min, t_max = periods
            if t_min <= (4 if discretization == "IM" else 2):
                raise ValueError(f"t_min {t_min} too short for {discretization}")
            T = torch.exp(torch.empty(P).uniform_(math.log(t_min), math.log(t_max)))
            theta, dt = 2 * math.pi / T, torch.sigmoid(self.steps.detach())
            A = (torch.tan(theta) / dt) ** 2 if discretization == "IM" else 2 * (1 - torch.cos(theta)) / dt ** 2
            with torch.no_grad():
                self.A_diag.copy_(A)

    def oscillators(self):
        return torch.relu(self.A_diag), torch.sigmoid(self.steps)

    def channel_kernel(self, L):
        """Real (H_out, H_in, L) kernel of the whole layer: K[k] = sum_p h_p[k] Re(C[:, p] B[p, :]).
        Exact because h is real and u is real: Re(C (h * (B u))) = (sum_p h_p Re(C_p B_p)) * u."""
        A, dt = self.oscillators()
        h = linoss_kernel(A, dt, L, self.disc)                                          # (P, L)
        W = torch.einsum("gp,ph->pgh", self.C[..., 0], self.B[..., 0]) - \
            torch.einsum("gp,ph->pgh", self.C[..., 1], self.B[..., 1])                  # (P, H, H)
        return torch.einsum("pk,pgh->ghk", h, W)

    def forward(self, u):
        """u (N, L, H) -> (N, L, H). On GPU: convolve with the H x H channel kernel, so no (N, 2P, L) state tensor
        is built (that tensor ran out of memory at 2,400 steps on a T4). On CPU the state path is ~4x faster, so it
        is used there. Both are exact (tests: <= 2.2e-6 vs the step-by-step recurrence, <= 6e-7 between paths)."""
        if not u.is_cuda:
            return self.forward_state_path(u)
        L = u.shape[1]
        K = self.channel_kernel(L)                                                      # (H, H, L)
        if L <= 64:
            idx = torch.arange(L, device=u.device)
            d = idx[:, None] - idx[None, :]                                             # l - j
            T = torch.where(d >= 0, K[:, :, d.clamp(min=0)], torch.zeros((), device=u.device, dtype=K.dtype))
            out = torch.einsum("njh,ghlj->nlg", u, T)                                   # T: (H, H, L, L)
        else:
            n = 2 * L                                                                   # >= 2L - 1: no wrap-around
            U = torch.fft.rfft(u, n=n, dim=1).permute(1, 0, 2)                          # (F, N, H_in)
            Kf = torch.fft.rfft(K, n=n).permute(2, 1, 0)                                # (F, H_in, H_out)
            out = torch.fft.irfft(torch.bmm(U, Kf).permute(1, 0, 2), n=n, dim=1)[:, :L]  # one matmul per frequency
        return out + u * self.D

    def forward_state_path(self, u):
        """CPU path: convolve each oscillator with its own kernel, then project with C. Used by forward() on CPU
        (about 4x faster there than the channel-kernel path); numerically equivalent to the GPU path (tests)."""
        A, dt = self.oscillators()
        h = linoss_kernel(A, dt, u.shape[1], self.disc)
        bu = torch.einsum("nlh,phc->ncpl", u, self.B)
        y = causal_conv(bu, h)
        out = torch.einsum("npl,hp->nlh", y[:, 0], self.C[..., 0]) - torch.einsum("npl,hp->nlh", y[:, 1], self.C[..., 1])
        return out + u * self.D

    @torch.no_grad()
    def frequencies_hz(self, fs):
        """Natural frequency of each oscillator for input sampled at fs (IM: theta = atan(dt sqrt(A)))."""
        A, dt = self.oscillators()
        th = torch.atan(dt * A.sqrt()) if self.disc == "IM" else torch.acos((1 - dt ** 2 * A / 2).clamp(-1, 1))
        return th * fs / (2 * math.pi)


def linoss_reference(layer, u):
    """Step-by-step recurrence of the official scan (complex state), for tests."""
    A, dt = layer.oscillators()
    M, f = _transition(A, dt, layer.disc)
    Bc = torch.complex(layer.B[..., 0], layer.B[..., 1])
    Cc = torch.complex(layer.C[..., 0], layer.C[..., 1])
    N, L, H = u.shape
    z = torch.zeros(N, A.shape[0], dtype=torch.complex64); y = torch.zeros_like(z)
    outs = []
    for t in range(L):
        bu = u[:, t].to(torch.complex64) @ Bc.T                                         # (N, P)
        z, y = M[:, 0, 0] * z + M[:, 0, 1] * y + f[:, 0] * bu, M[:, 1, 0] * z + M[:, 1, 1] * y + f[:, 1] * bu
        outs.append((y @ Cc.T).real + u[:, t] * layer.D)
    return torch.stack(outs, 1)



class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return self.w1(x) * torch.sigmoid(self.w2(x))

