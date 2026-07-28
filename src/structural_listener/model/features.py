"""Classical acoustic-emission feature extraction (Sprint 2.1).

The five features the AE literature actually uses for source classification
[Pollock], [Grosse], [Beattie]:

- **rise time**: onset \u2192 peak (s) \u2014 already in the hit detector.
- **ring-down count**: number of threshold crossings after the peak. The classic AE
  "counts" parameter: each oscillation that re-crosses the threshold is one count. A
  long-ringing burst (friction) has many; a short burst (crack) has few.
- **peak amplitude**: max |signal| in the hit window \u2014 already in the hit detector.
- **energy**: integral of signal^2 over the hit window (V\u00b2s or (m/s)\u00b2s).
  Impacts are high-energy; friction is sustained but lower-peak.
- **duration**: onset \u2192 end (s) \u2014 already in the hit detector.

These are the same features the Sprint 2.2 baseline (logistic regression + random
forest) consumes. The CNN (2.3) ignores them and works from the STFT \u2014 the
comparison is the interview story (2.4).

Per event, we extract features from the **strongest channel** (the channel with the
largest peak amplitude). This is a deliberate simplification: real AE systems fuse
features across channels, but for a 4-sensor corner array the strongest channel is a
stable, deterministic representative. The classifier sees one feature vector per event,
not four.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from structural_listener.dsp.hit_detection import HitDetectorConfig, detect_hits


@dataclass(frozen=True, slots=True)
class AeFeatures:
    """The five classical AE features for one event.

    Attributes:
        rise_time_s: time from onset to peak amplitude.
        ring_down_count: number of threshold re-crossings after the peak.
        peak_amplitude: max |signal| in the hit window (same units as the waveform).
        energy: integral of signal^2 over the hit window.
        duration_s: time the signal stayed above threshold (onset to end).
        channel: the sensor index these features came from (the strongest channel).

    """

    rise_time_s: float
    ring_down_count: int
    peak_amplitude: float
    energy: float
    duration_s: float
    channel: int

    def as_vector(self) -> np.ndarray:
        """Return the features as a float64 array in the documented order."""
        return np.array(
            [
                self.rise_time_s,
                float(self.ring_down_count),
                self.peak_amplitude,
                self.energy,
                self.duration_s,
            ],
            dtype=np.float64,
        )


FEATURE_NAMES: tuple[str, ...] = (
    "rise_time_s",
    "ring_down_count",
    "peak_amplitude",
    "energy",
    "duration_s",
)


def _ring_down_count(signal: np.ndarray, threshold: float, median: float, peak_idx: int) -> int:
    """Count threshold crossings after the peak.

    Each sample where |signal - median| crosses *upward* through threshold after the
    peak is one count. This is the classic AE "ring-down counts" parameter: a
    decaying oscillation that re-crosses the threshold N times has N counts.
    """
    if peak_idx >= signal.shape[0] - 1:
        return 0
    tail = np.abs(signal[peak_idx + 1 :] - median)
    # Upward crossings: tail > threshold and the previous sample was <= threshold.
    above = tail > threshold
    # Count transitions from False -> True (the start of each above-threshold run).
    # A run of length 1 is one count; a longer run is still one count (one crossing).
    crossings = 0
    prev = False
    for a in above:
        if a and not prev:
            crossings += 1
        prev = a
    return crossings


def extract_features(
    waveforms: np.ndarray,
    sample_rate_hz: float,
    config: HitDetectorConfig | None = None,
) -> AeFeatures | None:
    """Extract classical AE features from the strongest detected channel.

    Args:
        waveforms: (n_channels, n_samples) sensor signals.
        sample_rate_hz: sampling rate.
        config: hit-detector config (the same band/threshold the detector uses).

    Returns ``None`` if no channel produces a detected hit.

    The features come from the channel with the largest peak amplitude \u2014 the
    closest sensor to the source on a 4-corner array. This is a deterministic,
    explainable representative; real systems fuse channels, but that's a Sprint 2+
    refinement not needed for the baseline-vs-CNN comparison.

    """
    cfg = config or HitDetectorConfig()
    hits = detect_hits(waveforms, sample_rate_hz, cfg)
    if not hits:
        return None
    # Strongest channel = largest peak amplitude.
    hit = max(hits, key=lambda h: h.peak_amplitude)

    # Re-bandpass to get the signal the detector thresholded (same band).
    from structural_listener.dsp.tdoa import _bandpass

    bp = _bandpass(waveforms[hit.channel], sample_rate_hz, cfg.band_low_hz, cfg.band_high_hz)
    median, sigma = _noise_floor_for_threshold(bp, cfg)
    threshold = median + cfg.threshold_k * sigma

    onset = hit.onset_sample
    end = onset + max(int(hit.duration_s * sample_rate_hz), 1)
    end = min(end, bp.shape[0])
    segment = bp[onset:end]
    peak_rel = int(np.argmax(np.abs(segment)))
    peak_idx = onset + peak_rel

    ring_down = _ring_down_count(bp, threshold, median, peak_idx)
    energy = float(np.sum(segment**2) / sample_rate_hz)

    return AeFeatures(
        rise_time_s=hit.rise_time_s,
        ring_down_count=ring_down,
        peak_amplitude=hit.peak_amplitude,
        energy=energy,
        duration_s=hit.duration_s,
        channel=hit.channel,
    )


def _noise_floor_for_threshold(wave_bp: np.ndarray, cfg: HitDetectorConfig) -> tuple[float, float]:
    """Reproduce the hit detector's noise floor so the threshold matches.

    The hit detector computes (median, sigma) before bandpassing; here we have the
    bandpassed signal already, so we re-estimate on it. This keeps the ring-down
    threshold consistent with the detection threshold.
    """
    skip = 20
    n = min(cfg.noise_window_samples, wave_bp.shape[0] - skip)
    if n <= 0:
        return 0.0, float(np.std(wave_bp)) or 1.0e-15
    quiet = wave_bp[skip : skip + n]
    sigma_pre = float(np.std(quiet))
    if sigma_pre > 1.0e-15:
        return float(np.median(quiet)), sigma_pre
    # Fall back to P75 if the pre-arrival window is degenerate.
    median_global = float(np.median(wave_bp))
    p75 = float(np.percentile(np.abs(wave_bp), 75))
    sigma_global = max((p75 - abs(median_global)) / 0.674, 1.0e-15)
    return median_global, sigma_global
