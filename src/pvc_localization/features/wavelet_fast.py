"""Batched CWT scalogram, numerically equivalent to extract_cwt_scalogram (pywt.cwt, method="fft"), on CPU or GPU.

Same wavelet, scales, sampled wavelet integral (pywt.integrate_wavelet, precision 12 as in pywt.cwt), convolution, differencing,
trimming, 8-sample averaging and log1p as features/wavelet.py. Two changes affect only the computation:

1. Kernel truncation. pywt convolves each beat (N = 1600 samples) with the full sampled wavelet of each scale (up
   to 16 001 samples at 2 Hz) and then keeps N + 1 central outputs. Only kernel samples within N of those outputs
   can contribute, so each kernel is cut to at most 2N samples. This is exact (the removed products are never
   used), and the FFT length drops from about 17 600 to 4 800.
2. Batching. All beats, leads and scales are transformed in one FFT call (one kernel spectrum per scale, computed
   once), in PyTorch on the CPU or the GPU.

Only rounding differs from pywt (float64 by default; float32 on request).
"""
import math

import numpy as np
import pywt
import torch
from scipy.fft import next_fast_len

from pvc_localization import config


class FastCWT:
    def __init__(self, n_samples: int = config.BEAT_LENGTH_SAMPLES, wavelet: str = config.CWT_WAVELET,
                 n_scales: int = config.CWT_NUM_SCALES, fs: int = config.SAMPLING_RATE_HZ,
                 device: str = "cpu", dtype: torch.dtype = torch.float64, input_dtype=np.float64):
        # input_dtype: dtype of the beats given to pywt; pywt samples the wavelet in that precision, so the kernels
        # are built the same way (the FFTs themselves run in `dtype`)
        w = pywt.ContinuousWavelet(wavelet)
        freqs = np.geomspace(config.CWT_FMIN_HZ, config.FEATURE_FMAX_HZ, n_scales)
        scales = pywt.central_frequency(wavelet) * fs / freqs
        int_psi, x = pywt.integrate_wavelet(w, precision=12)  # pywt.cwt default precision
        real = np.dtype(input_dtype)
        cplx = np.result_type(real, np.complex64)
        int_psi = np.asarray(np.conj(int_psi), dtype=cplx)
        x = np.asarray(x, dtype=real)
        n = n_samples
        kernels, starts = [], []
        for scale in scales:  # as in pywt.cwt
            step = x[1] - x[0]
            j = (np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)).astype(int)
            j = j[j < int_psi.size]
            h = int_psi[j][::-1]
            length = h.size
            s = math.floor((length - 2) / 2)  # first kept output of the full convolution (before np.diff)
            a, b = max(0, s - n + 1), min(length - 1, s + n)  # kernel samples that reach outputs s .. s + n
            kernels.append(h[a:b + 1]); starts.append(s - a)
        self.m = next_fast_len(n + max(k.size for k in kernels) - 1)
        kf = np.stack([np.fft.fft(k, self.m) for k in kernels])  # (S, M)
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.device, self.dtype = device, dtype
        self.kf = torch.tensor(kf, dtype=cdtype, device=device)
        self.idx = torch.tensor(np.array(starts)[:, None] + np.arange(n + 1)[None, :], dtype=torch.long,
                                device=device)  # (S, N+1); int64, as gather requires (numpy < 2 gives int32 on Windows)
        self.sqrt_scale = torch.tensor(np.sqrt(scales), dtype=dtype, device=device)[:, None]
        self.n, self.ds = n, config.CWT_TIME_DOWNSAMPLE

    @torch.no_grad()
    def __call__(self, beats, chunk: int = 8) -> torch.Tensor:
        """beats: (B, 12, N) array or tensor -> (B, 12, n_scales, N // 8) log-magnitude scalograms on the device."""
        x = torch.as_tensor(beats, dtype=self.dtype).to(self.device)
        out = []
        for k in range(0, x.shape[0], chunk):
            xb = x[k:k + chunk]
            b, leads, _ = xb.shape
            xf = torch.fft.fft(xb, self.m, dim=-1)  # (b, 12, M)
            conv = torch.fft.ifft(xf[:, :, None, :] * self.kf[None, None], dim=-1)  # (b, 12, S, M)
            conv = torch.gather(conv, -1, self.idx[None, None].expand(b, leads, -1, -1))  # (b, 12, S, N+1)
            coef = -self.sqrt_scale * torch.diff(conv, dim=-1)  # (b, 12, S, N)
            mag = coef.abs()
            t = self.n // self.ds
            mag = mag[..., :t * self.ds].reshape(b, leads, mag.shape[2], t, self.ds).mean(-1)
            out.append(torch.log1p(mag))
        return torch.cat(out)
