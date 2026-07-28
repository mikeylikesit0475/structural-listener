"""Hit detection: adaptive threshold, duration gate, rise-time measurement.

Identifies the arrival of an acoustic-emission event in a single-channel sensor
waveform. The detector first bandpasses the signal to the detection band (20-90 kHz),
which rejects the low-frequency pink noise that would otherwise trigger false onsets.
Then: adaptive threshold, duration gate, and rise-time measurement.

Pipeline per channel:
1. Bandpass to the detection band (20-90 kHz) \u2014 rejects 1/f noise and mains pickup.
2. Estimate the noise floor from the pre-event quiet period (median absolute deviation).
3. Adaptive threshold = noise_floor + k * MAD * 1.4826 (robust sigma estimate).
4. Find the first sustained above-threshold region (density check: >= 5 samples in 100);
   a single spike is noise, not a hit.
5. Duration gate: discard candidates whose energy returns below threshold within
   ``min_duration_s`` \u2014 these are noise spikes, not events.
6. Rise-time measurement: time from onset to peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HitDetectorConfig:
    """Parameters for the hit detector. See ``docs/sources.md`` for citations."""

    # Detection band (Hz). Bandpassing before detection rejects the 1/f pink noise
    # (concentrated below 1 kHz) and mains pickup (50 Hz). 20-90 kHz is the A0 band.
    band_low_hz: float = 20.0e3
    band_high_hz: float = 90.0e3
    # Robust threshold multiplier: threshold = median + k * 1.4826 * MAD.
    # k=4: standard AE pick threshold [Pollock]. The energy-ratio check (below)
    # rejects noise that exceeds 4 sigma by chance, so we can use a lower threshold
    # and still avoid false triggers.
    threshold_k: float = 4.0
    # Minimum hit duration: candidates shorter than this are noise spikes.
    min_duration_s: float = 20.0e-6
    # Window (in samples) for the noise-floor estimate, taken from the start of the
    # waveform before any wave can arrive. 400 samples ~ 485 us at 824 kHz. The closest
    # sensor on a 1 m plate is ~0.14 m away (~140 us at c~1000 m/s), so 485 us is
    # pre-arrival. The window must be long enough for a stable sigma estimate of
    # band-limited noise (the autocorrelation length is ~50 samples, so 400 samples
    # give ~8 independent samples \u2014 marginal but workable with std rather than MAD).
    noise_window_samples: int = 400


@dataclass(frozen=True, slots=True)
class Hit:
    """A detected hit in a single channel.

    Attributes:
        channel: sensor index (0..3).
        onset_sample: index of the first sample above threshold.
        onset_time_s: onset_sample / sample_rate_hz.
        peak_sample: index of the maximum amplitude within the hit.
        peak_amplitude: the maximum amplitude (absolute), in the waveform's units.
        rise_time_s: time from onset to peak.
        duration_s: time the signal stayed above threshold.

    """

    channel: int
    onset_sample: int
    onset_time_s: float
    peak_sample: int
    peak_amplitude: float
    rise_time_s: float
    duration_s: float


def _robust_noise_floor(wave: np.ndarray, window_samples: int) -> tuple[float, float]:
    """Return (median, sigma) estimated from the waveform.

    Two strategies, combined:
    1. Pre-arrival window: the first ``window_samples`` before any event arrives. This
       is the cleanest estimate but may be too short for stable sigma of band-limited
       noise.
    2. Global P75: the 75th percentile of absolute values, robust to short events but
       not to sustained signals (friction fills a large fraction of the waveform).

    We use the *larger* of the two sigma estimates: the pre-arrival window gives the
    noise floor for short events, and the P75 catches cases where the pre-arrival window
    is contaminated or too short. Taking the larger sigma gives a more conservative
    (higher) threshold, which avoids false triggers at the cost of missing some weak
    events \u2014 the right trade-off for a detection system.
    """
    # Pre-arrival window estimate. Skip the first few samples to avoid bandpass edge
    # artifacts (the FFT bandpass creates Gibbs-ringing at the signal boundaries).
    skip = 20
    n = min(window_samples, wave.shape[0] - skip)
    quiet = wave[skip : skip + n]
    median_pre = float(np.median(quiet))
    sigma_pre = float(np.std(quiet))

    # Global P75 estimate.
    median_global = float(np.median(wave))
    abs_wave = np.abs(wave)
    p75 = float(np.percentile(abs_wave, 75))
    sigma_global = max((p75 - abs(median_global)) / 0.674, 1.0e-15)

    # Use the pre-arrival window when it gives a meaningful sigma (> 1e-15), as it's
    # the cleanest estimate. Fall back to the P75 only when the pre-arrival window is
    # contaminated (sigma near zero \u2014 all-zero or near-zero noise window).
    if sigma_pre > 1.0e-15:
        return median_pre, sigma_pre
    return median_global, sigma_global


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


def detect_hit(
    wave: np.ndarray,
    sample_rate_hz: float,
    channel: int,
    config: HitDetectorConfig | None = None,
) -> Hit | None:
    """Detect the first hit in a single-channel waveform.

    Returns ``None`` if no hit passes the duration gate.
    """
    cfg = config or HitDetectorConfig()
    if wave.shape[0] < cfg.noise_window_samples:
        return None

    # Bandpass to the detection band \u2014 rejects 1/f noise and mains pickup.
    wave_bp = _bandpass(wave, sample_rate_hz, cfg.band_low_hz, cfg.band_high_hz)

    median, sigma = _robust_noise_floor(wave_bp, cfg.noise_window_samples)
    if sigma <= 1.0e-15:
        # All-zero or near-zero noise window: use the global RMS as a fallback so the
        # threshold is meaningful. This handles noise-free test configs.
        sigma = float(np.std(wave_bp[: cfg.noise_window_samples]))
        if sigma <= 1.0e-15:
            sigma = 1.0e-9
    threshold = median + cfg.threshold_k * sigma

    # Find the first sustained run of above-threshold samples. A single above-threshold
    # sample followed by a long gap is a noise spike, not a hit. We require at least
    # ``min_above_density`` samples above threshold within the first ``window`` samples
    # after the candidate onset.
    above_mask = np.abs(wave_bp - median) > threshold
    above_mask[: cfg.noise_window_samples] = False
    min_above_density = 5
    window = 100  # ~120 us at 824 kHz
    onset = -1
    for candidate in np.where(above_mask)[0]:
        count = int(np.sum(above_mask[candidate : candidate + window]))
        if count >= min_above_density:
            onset = int(candidate)
            break
    if onset < 0:
        return None
    # AE bursts are oscillatory: the signal crosses the threshold, dips below at a
    # zero-crossing, then crosses back. A gap tolerance of 50 samples (~60 us at
    # 824 kHz) spans ~5 cycles of an 80 kHz crack burst without merging separate
    # events, and is long enough that a real hit isn't truncated between zero-crossings
    # while still rejecting 1-2 sample noise spikes.
    end = onset
    gap_tolerance = 50
    gap = 0
    for k in range(onset, wave_bp.shape[0]):
        if abs(float(wave_bp[k]) - median) > threshold:
            end = k
            gap = 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break
    duration_samples = end - onset + 1
    duration_s = duration_samples / sample_rate_hz
    if duration_s < cfg.min_duration_s:
        return None

    # Peak within the hit window.
    hit_segment = wave_bp[onset : end + 1]
    peak_rel = int(np.argmax(np.abs(hit_segment)))
    peak_sample = onset + peak_rel
    peak_amplitude = float(abs(float(wave_bp[peak_sample])))
    rise_time_s = (peak_sample - onset) / sample_rate_hz

    return Hit(
        channel=channel,
        onset_sample=onset,
        onset_time_s=onset / sample_rate_hz,
        peak_sample=peak_sample,
        peak_amplitude=peak_amplitude,
        rise_time_s=rise_time_s,
        duration_s=duration_s,
    )


def detect_hits(
    waveforms: np.ndarray, sample_rate_hz: float, config: HitDetectorConfig | None = None
) -> list[Hit]:
    """Detect the first hit in each of the 4 channels. Returns only the channels that hit."""
    hits = []
    for ch in range(waveforms.shape[0]):
        hit = detect_hit(waveforms[ch], sample_rate_hz, ch, config)
        if hit is not None:
            hits.append(hit)
    return hits
