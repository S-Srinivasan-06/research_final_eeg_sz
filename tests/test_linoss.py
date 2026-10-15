"""LinOSS layer: the convolution paths equal the step-by-step recurrence of the official scan (Rusch & Rus 2025)."""
import torch

from szfusion.linoss.layer import LinOSSLayer, linoss_reference


def channel_kernel_path(layer, u):
    """The GPU branch of LinOSSLayer.forward, evaluated on CPU (H x H channel kernel, one matmul per frequency)."""
    L = u.shape[1]
    K = layer.channel_kernel(L)
    n = 2 * L
    U = torch.fft.rfft(u, n=n, dim=1).permute(1, 0, 2)
    Kf = torch.fft.rfft(K, n=n).permute(2, 1, 0)
    return torch.fft.irfft(torch.bmm(U, Kf).permute(1, 0, 2), n=n, dim=1)[:, :L] + u * layer.D


def test_paths_match_recurrence():
    torch.manual_seed(0)
    for disc, per in (("IM", None), ("IM", (5, 240)), ("IMEX", None)):
        layer = LinOSSLayer(8, 6, disc, per).double()
        for L in (40, 130):                                   # direct (<= 64) and FFT branches of causal_conv
            u = torch.randn(3, L, 6, dtype=torch.float64)
            ref = linoss_reference(layer.float(), u.float()).double(); layer.double()
            assert torch.allclose(layer.forward_state_path(u), ref, atol=1e-4)
            assert torch.allclose(channel_kernel_path(layer, u), layer.forward_state_path(u), atol=1e-9)


def test_period_initialisation():
    layer = LinOSSLayer(512, 4, "IM", (5, 240))
    periods = 1 / layer.frequencies_hz(1.0)
    assert periods.min() >= 5 - 1e-3 and periods.max() <= 240 + 1e-2
