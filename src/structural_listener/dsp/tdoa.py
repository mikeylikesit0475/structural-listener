"""Time-difference-of-arrival (TDOA) estimation.

Two pickers are provided:

1. **Gated envelope-peak picker** (`pick_arrival_times`): the primary arrival-time
   estimator used by the Sprint 1 localisation pipeline. For each channel, bandpass
   to the fixed detection band, take the analytic-signal envelope, find the first
   sample crossing `peak_fraction` of the envelope maximum, then return the envelope
   peak within a short `gate_s` window after that crossing. The gate excludes later
   boundary reflections (ADR-006), and the *peak* (not the first-crossing) is robust
   to band-limited noise bursts that fire the threshold before the real wave arrives.
   This yields one absolute arrival time per channel; the multilateration solver takes
   differences.

2. **Onset + cross-correlation refinement** (`estimate_tdoa`, `estimate_all_tdoas`):
   a pairwise estimator kept for the synthetic-delay unit tests and as a fallback.
   On a reflective dispersive plate the cross-correlation window can catch a
   reflection instead of the direct arrival; the gated envelope-peak picker is the
   preferred method for end-to-end localisation.

The fixed detection band (20-90 kHz) is used for all sensors so the dispersion bias
cancels in the TDOA differences (ADR-003).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TdoaConfig:
    """Parameters for TDOA estimation. See ``docs/sources.md`` and ADR-003."""

    # Fixed detection band (Hz). All sensors use the same band so the dispersion bias
    # cancels in the TDOA differences. 20-90 kHz is the well-resolved A0 range at dx=5mm.
    band_low_hz: float = 20.0e3
    band_high_hz: float = 90.0e3
    # Window around the hit onset for cross-correlation refinement (seconds). Must be
    # long enough to capture the direct burst (~0.3-0.5 ms) but short enough to exclude
    # the first boundary reflection (~1-2 ms on a 1 m plate).
    refinement_window_s: float = 0.5e-3
    # --- Gated envelope-peak picker (Sprint 1.3) ---
    # Fraction of the envelope maximum used as the first-crossing threshold. Low
    # enough to fire on the direct arrival, high enough to ignore band-limited noise
    # bursts. 0.3 empirically separates the direct burst from noise on the 1 m plate.
    peak_fraction: float = 0.3
    # Length of the gate (seconds) after the first crossing within which we take the
    # envelope peak. Must be shorter than the direct-to-first-reflection gap on the
    # plate (~1 ms on a 1 m plate); 0.35 ms is conservative.
    gate_s: float = 0.35e-3


@dataclass(frozen=True, slots=True)
class TdoaEstimate:
    """A single TDOA estimate between two channels.

    Attributes:
        lag_samples: integer-sample lag (test - ref, positive = test arrives later).
        lag_s: sub-sample-refined lag in seconds.
        confidence: normalised correlation peak height in [0, 1].

    """

    lag_samples: int
    lag_s: float
    confidence: float


def _bandpass(
    signal: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float
) -> np.ndarray:
    """FFT bandpass filter: zero bins outside [low_hz, high_hz]."""
    n = signal.shape[0]
    spec = np.fft.rfft(signal * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec_band = spec * mask
    out: np.ndarray = np.fft.irfft(spec_band, n=n)
    return out


def _envelope(signal: np.ndarray) -> np.ndarray:
    """Analytic-signal envelope via Hilbert transform (FFT implementation)."""
    n = signal.shape[0]
    spec = np.fft.fft(signal, n=n)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spec * h)
    return np.abs(analytic)


def _parabolic_subsample_peak(corr: np.ndarray, peak_idx: int) -> float:
    """Parabolic interpolation around a 3-point correlation peak for sub-sample lag.

    Returns the x-offset of the vertex in [-0.5, +0.5] samples.
    """
    if peak_idx <= 0 or peak_idx >= corr.shape[0] - 1:
        return 0.0
    y_m = float(corr[peak_idx - 1])
    y_0 = float(corr[peak_idx])
    y_p = float(corr[peak_idx + 1])
    denom = y_m - 2.0 * y_0 + y_p
    if abs(denom) < 1e-30:
        return 0.0
    return 0.5 * (y_m - y_p) / denom


def estimate_tdoa(
    ref_signal: np.ndarray,
    test_signal: np.ndarray,
    sample_rate_hz: float,
    config: TdoaConfig | None = None,
    ref_onset_sample: int = 0,
    test_onset_sample: int = 0,
) -> TdoaEstimate | None:
    """Estimate the TDOA ``test - ref`` (positive = test arrives later).

    Uses the hit-detector onsets for the integer-sample TDOA, then refines with
    cross-correlation of the band-limited envelopes in a short window around the onsets.

    Args:
        ref_signal: the reference channel waveform.
        test_signal: the test channel waveform.
        sample_rate_hz: sampling rate of both signals.
        config: TDOA parameters. If None, uses ``TdoaConfig()`` defaults.
        ref_onset_sample: onset sample of the hit in the reference channel.
        test_onset_sample: onset sample of the hit in the test channel.

    Returns ``None`` if the onsets are missing (both zero) or the correlation is too weak.

    """
    cfg = config or TdoaConfig()

    # Integer-sample TDOA from the hit-detector onsets. This is the primary estimate:
    # the onset is the first above-threshold crossing (the direct arrival), robust to
    # reflections (ADR-006). At 824.5 kHz, 1 sample = 1.2 us = ~1.2 mm at c=1000 m/s,
    # well within Sprint 1's < 5 cm target. Sub-sample refinement is attempted but
    # falls back to the integer-sample estimate if the correlation is inconsistent.
    if ref_onset_sample == 0 and test_onset_sample == 0:
        return None
    onset_lag_samples = test_onset_sample - ref_onset_sample
    onset_lag_s = onset_lag_samples / sample_rate_hz

    # Sub-sample refinement via envelope cross-correlation in a window around the onsets.
    window = int(cfg.refinement_window_s * sample_rate_hz)
    start = max(min(ref_onset_sample, test_onset_sample) - 10, 0)
    end = min(start + window, ref_signal.shape[0], test_signal.shape[0])
    if end - start < 20:
        return TdoaEstimate(lag_samples=onset_lag_samples, lag_s=onset_lag_s, confidence=0.0)

    ref_bp = _bandpass(ref_signal, sample_rate_hz, cfg.band_low_hz, cfg.band_high_hz)
    test_bp = _bandpass(test_signal, sample_rate_hz, cfg.band_low_hz, cfg.band_high_hz)
    ref_env = _envelope(ref_bp[start:end])
    test_env = _envelope(test_bp[start:end])

    # Cross-correlate the envelopes; search a small range around the onset-based lag.
    corr = np.correlate(ref_env, test_env, mode="full")
    centre = ref_env.shape[0] - 1
    # The onset-based lag in the cropped frame's coordinate system.
    ref_offset = ref_onset_sample - start
    test_offset = test_onset_sample - start
    expected_peak = centre + (test_offset - ref_offset)
    # Search +/- 20 samples around the expected peak.
    search_radius = 20
    lo = max(expected_peak - search_radius, 0)
    hi = min(expected_peak + search_radius + 1, corr.shape[0])
    if lo >= hi:
        return TdoaEstimate(lag_samples=onset_lag_samples, lag_s=onset_lag_s, confidence=0.0)
    peak_idx = lo + int(np.argmax(corr[lo:hi]))
    peak_value = float(corr[peak_idx])

    ref_energy = float(np.sqrt(np.sum(ref_env**2)))
    test_energy = float(np.sqrt(np.sum(test_env**2)))
    norm = ref_energy * test_energy
    confidence = peak_value / norm if norm > 0.0 else 0.0

    raw_lag = peak_idx - centre
    # Only accept the refined lag if it's within 5 samples of the onset-based lag;
    # otherwise the correlation found a reflection or noise, and we fall back.
    if abs(raw_lag - onset_lag_samples) > 5:
        return TdoaEstimate(lag_samples=onset_lag_samples, lag_s=onset_lag_s, confidence=confidence)
    sub_offset = _parabolic_subsample_peak(corr, peak_idx)
    refined_lag = raw_lag + sub_offset
    lag_s = refined_lag / sample_rate_hz

    return TdoaEstimate(
        lag_samples=raw_lag,
        lag_s=lag_s,
        confidence=confidence,
    )


def estimate_all_tdoas(
    waveforms: np.ndarray,
    sample_rate_hz: float,
    ref_channel: int = 0,
    config: TdoaConfig | None = None,
    onsets_samples: dict[int, int] | None = None,
) -> dict[int, TdoaEstimate]:
    """Estimate TDOA of each channel relative to ``ref_channel``.

    Args:
        waveforms: (n_channels, n_samples) sensor signals.
        sample_rate_hz: sampling rate.
        ref_channel: the reference channel index.
        config: TDOA parameters.
        onsets_samples: {channel: onset_sample} from the hit detector. Required for
            robust TDOA \u2014 without onsets, reflections corrupt the correlation.

    Returns a dict {channel: TdoaEstimate} for channels that produced a valid estimate.

    """
    result: dict[int, TdoaEstimate] = {}
    ref = waveforms[ref_channel]
    ref_onset = onsets_samples.get(ref_channel, 0) if onsets_samples else 0
    for ch in range(waveforms.shape[0]):
        if ch == ref_channel:
            continue
        test_onset = onsets_samples.get(ch, 0) if onsets_samples else 0
        est = estimate_tdoa(
            ref,
            waveforms[ch],
            sample_rate_hz,
            config,
            ref_onset_sample=ref_onset,
            test_onset_sample=test_onset,
        )
        if est is not None:
            result[ch] = est
    return result


def _gated_envelope_peak(signal: np.ndarray, sample_rate_hz: float, cfg: TdoaConfig) -> float:
    """Return the absolute arrival time (seconds) of the direct wave at one channel.

    Steps:
    1. Bandpass to the fixed detection band.
    2. Take the analytic-signal envelope (Hilbert transform via FFT).
    3. Find the first sample where the envelope exceeds ``peak_fraction`` of its
       global maximum \u2014 the direct-arrival window opens here.
    4. Return the time of the envelope peak within the ``gate_s`` window after that
       crossing. The gate excludes later boundary reflections (ADR-006).

    Returns NaN if no sample crosses the threshold (no detectable arrival).
    """
    bp = _bandpass(signal, sample_rate_hz, cfg.band_low_hz, cfg.band_high_hz)
    env = _envelope(bp)
    peak = float(np.max(env))
    if peak <= 0.0:
        return float("nan")
    above = np.where(env >= cfg.peak_fraction * peak)[0]
    if above.size == 0:
        return float("nan")
    start = int(above[0])
    gate_end = min(start + int(cfg.gate_s * sample_rate_hz), env.shape[0])
    if gate_end <= start:
        return float(start) / sample_rate_hz
    peak_idx = start + int(np.argmax(env[start:gate_end]))
    return float(peak_idx) / sample_rate_hz


def pick_arrival_times(
    waveforms: np.ndarray,
    sample_rate_hz: float,
    config: TdoaConfig | None = None,
) -> np.ndarray:
    """Pick one absolute arrival time per channel via the gated envelope-peak method.

    Args:
        waveforms: (n_channels, n_samples) sensor signals.
        sample_rate_hz: sampling rate.
        config: picker parameters. If None, uses ``TdoaConfig()`` defaults.

    Returns:
        (n_channels,) array of absolute arrival times in seconds. Channels with no
        detectable arrival are NaN; the multilateration solver excludes them.

    """
    cfg = config or TdoaConfig()
    times = np.empty(waveforms.shape[0], dtype=np.float64)
    for ch in range(waveforms.shape[0]):
        times[ch] = _gated_envelope_peak(waveforms[ch], sample_rate_hz, cfg)
    return times
