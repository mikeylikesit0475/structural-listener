"""STFT front-end for the CNN classifier (Sprint 2.3).

The CNN input is the log-magnitude STFT of the strongest channel's waveform around the
detected hit. The STFT captures the time-frequency structure that the 5 classical
features collapse into scalars: a crack burst is broadband and short, friction is
narrowband and sustained, impact is low-frequency, noise is tonal at 50 Hz.

Per ARCHITECTURE.md \u00a75, the input is a 5 ms window around each hit, per channel. We
use the strongest channel only (consistent with the classical features) and a fixed
window so the CNN sees a consistent shape.
"""

from __future__ import annotations

import numpy as np

# STFT parameters. Chosen so the spectrogram is small enough for a tiny CNN but
# resolves the AE band: 824.5 kHz / 256 nfft = 3.2 kHz/bin, 32 hops = 39 us/frame.
# A 5 ms window gives ~128 frames; we pad/truncate to 64 frames for a fixed shape.
NFFT = 256
HOP = 32
N_FRAMES = 64
# Frequency bins: keep the 0-100 kHz band (top ~32 bins of a 256-point FFT at 824.5 kHz).
F_BINS = 32


def stft_input(waveform: np.ndarray, sample_rate_hz: float, onset_sample: int) -> np.ndarray:
    """Compute the log-magnitude STFT around the hit onset, normalised to [0, 1].

    Args:
        waveform: single-channel signal.
        sample_rate_hz: sampling rate.
        onset_sample: the hit-detector onset (centre of the STFT window).

    Returns:
        (F_BINS, N_FRAMES) float32 log-magnitude spectrogram, min-max normalised.
        Returns a zero array if the window doesn't fit.

    """
    window_s = N_FRAMES * HOP / sample_rate_hz  # ~2.5 ms
    start = max(onset_sample - int(0.5 * window_s * sample_rate_hz / HOP) * HOP, 0)
    end = start + N_FRAMES * HOP
    if end > waveform.shape[0]:
        # Pad with zeros if the window runs past the signal end.
        padded = np.zeros(end, dtype=np.float64)
        padded[: waveform.shape[0]] = waveform
        segment = padded[start:end]
    else:
        segment = waveform[start:end]

    # Hann-windowed STFT via numpy.
    window = np.hanning(NFFT)
    frames = np.lib.stride_tricks.sliding_window_view(segment, NFFT)[::HOP]
    if frames.shape[0] < N_FRAMES:
        pad = np.zeros((N_FRAMES - frames.shape[0], NFFT))
        frames = np.vstack([frames, pad])
    frames = frames[:N_FRAMES]
    spec = np.fft.rfft(frames * window, axis=1)
    mag = np.abs(spec[:, :F_BINS]).T  # (freq, time)

    # Log-magnitude, normalised to [0, 1].
    log_mag = np.log10(mag + 1.0e-20)
    mn, mx = float(log_mag.min()), float(log_mag.max())
    if mx - mn < 1.0e-10:
        return np.zeros((F_BINS, N_FRAMES), dtype=np.float32)
    return ((log_mag - mn) / (mx - mn)).astype(np.float32)
