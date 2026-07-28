"""Hit detection: adaptive threshold, duration gate, rise-time measurement.

Identifies the arrival of an acoustic-emission event in a single-channel sensor
waveform. The detector is deliberately classical \u2014 this is the AE literature's standard
approach [Pollock], [Grosse] and is the same path the Sprint 2 classifier's features will
be built on.

Pipeline per channel:
1. Estimate the noise floor from the pre-event quiet period (median absolute deviation).
2. Adaptive threshold = noise_floor + k * MAD * 1.4826 (robust sigma estimate).
3. Find the first sample above threshold; that's the candidate hit onset.
4. Duration gate: discard candidates whose energy returns below threshold within
   ``min_duration_s`` \u2014 these are noise spikes, not events.
5. Rise-time measurement: time from onset to peak, in microseconds. This is a Sprint 2
   classifier feature and a sanity check (real AE hits rise in us-ms, not instantaneous).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HitDetectorConfig:
    """Parameters for the hit detector. See ``docs/sources.md`` for citations."""

    # Robust threshold multiplier: threshold = median + k * 1.4826 * MAD.
    # k=4 gives ~4 sigma above the noise floor \u2014 standard AE pick threshold [Pollock]
    # (3-5 sigma). The 1/f pink noise component has heavier tails than Gaussian, so 3
    # sigma produces false triggers; 4 is the safe choice. On a 1 m plate with realistic
    # noise, the far channels may not detect a centre-plate event \u2014 that's physically
    # honest (geometric spreading 1/sqrt(r) + material attenuation), and the localisation
    # solver handles 3-channel cases with a degraded confidence.
    threshold_k: float = 4.0
    # Minimum hit duration: candidates shorter than this are noise spikes.
    # 20 us is the shortest physical AE burst we model (crack growth at 80 kHz, ~0.3 ms).
    min_duration_s: float = 20.0e-6
    # Window (in samples) for the noise-floor estimate, taken from the very start of the
    # waveform before any wave can arrive. 50 samples ~ 60 us at 824 kHz; the closest
    # sensor on a 1 m plate is ~0.14 m away, giving ~140 us at c~1000 m/s \u2014 so 60 us
    # is safely pre-arrival even for the nearest sensor.
    noise_window_samples: int = 50


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
    """Return (median, robust_sigma) estimated from the first ``window_samples`` of ``wave``.

    The robust sigma is 1.4826 * MAD (median absolute deviation), which is the
    consistent estimator for Gaussian noise and is insensitive to the event itself
    as long as the window is pre-arrival.
    """
    n = min(window_samples, wave.shape[0])
    quiet = wave[:n]
    median = float(np.median(quiet))
    mad = float(np.median(np.abs(quiet - median)))
    robust_sigma = 1.4826 * mad
    return median, robust_sigma


def detect_hit(
    wave: np.ndarray,
    sample_rate_hz: float,
    channel: int,
    config: HitDetectorConfig | None = None,
) -> Hit | None:
    """Detect the first hit in a single-channel waveform.

    Returns ``None`` if no hit passes the duration gate \u2014 e.g. a pure-noise channel or
    a sensor that received nothing above threshold.
    """
    cfg = config or HitDetectorConfig()
    if wave.shape[0] < cfg.noise_window_samples:
        return None

    median, sigma = _robust_noise_floor(wave, cfg.noise_window_samples)
    if sigma <= 0.0:
        # All-zero noise window: fall back to a tiny sigma so the threshold is just
        # the median. This handles the noise-free test configs.
        sigma = 1.0e-30
    threshold = median + cfg.threshold_k * sigma

    # Find the first sustained run of above-threshold samples. A single above-threshold
    # sample followed by a long gap is a noise spike, not a hit. We require at least
    # ``min_above_density`` samples above threshold within the first ``window`` samples
    # after the candidate onset.
    above_mask = np.abs(wave - median) > threshold
    above_mask[: cfg.noise_window_samples] = False
    min_above_density = 5
    window = 100  # ~120 us at 824 kHz \u2014 one full 80 kHz burst is ~25 samples
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
    for k in range(onset, wave.shape[0]):
        if abs(float(wave[k]) - median) > threshold:
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
    hit_segment = wave[onset : end + 1]
    peak_rel = int(np.argmax(np.abs(hit_segment)))
    peak_sample = onset + peak_rel
    peak_amplitude = float(abs(float(wave[peak_sample])))
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
