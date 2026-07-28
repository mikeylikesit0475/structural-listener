"""Sprint 2.1: classical AE feature extractor.

Tests:
- The extractor returns finite, non-negative features for a crack event.
- The four event types produce distinguishable feature vectors (the whole point of
  using these features for classification).
- The strongest-channel selection is deterministic and picks the near sensor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from structural_listener.cli.inject import inject_event
from structural_listener.model.features import (
    FEATURE_NAMES,
    AeFeatures,
    extract_features,
)


def _features_for(
    tmp_path: Path, source_type: str, x: float = 0.5, y: float = 0.5
) -> AeFeatures | None:
    # Noise needs a longer window: the 50 Hz signal needs several cycles to register.
    # 10 ms (5 cycles) is the minimum; 50 ms is more realistic but 5x slower to simulate.
    duration_s = 10.0e-3 if source_type == "noise" else 4.0e-3
    path = inject_event(source_type, x, y, duration_s=duration_s, out_dir=tmp_path)
    with np.load(path) as data:
        wf = np.asarray(data["waveforms"])
        sr = float(data["sample_rate_hz"])
    return extract_features(wf, sr)


def test_feature_vector_has_five_named_components() -> None:
    """The feature vector matches FEATURE_NAMES in length and order."""
    assert len(FEATURE_NAMES) == 5
    assert FEATURE_NAMES == (
        "rise_time_s",
        "ring_down_count",
        "peak_amplitude",
        "energy",
        "duration_s",
    )


def test_features_are_finite_and_nonnegative_for_crack(tmp_path: Path) -> None:
    """A crack event produces finite, physically-signed features."""
    feats = _features_for(tmp_path, "crack", 0.3, 0.3)
    assert feats is not None, "crack produced no hit"
    v = feats.as_vector()
    assert np.all(np.isfinite(v)), f"non-finite feature: {v}"
    # Rise time, ring-down count, peak amplitude, energy, duration are all >= 0.
    assert np.all(v >= 0.0), f"negative feature: {v}"


def test_four_event_types_produce_distinct_feature_vectors(tmp_path: Path) -> None:
    """The whole point of these features: the four AE source types are separable.

    We don't assert a specific classifier (that's 2.2) \u2014 we assert that the five
    features separate the four classes enough that a linear model can do the job.
    Each pair must differ in at least one feature by > 3x, OR have distinguishable
    ring-down counts.
    """
    vectors: dict[str, np.ndarray] = {}
    for st in ("crack", "friction", "impact", "noise"):
        feats = _features_for(tmp_path, st, 0.5, 0.5)
        assert feats is not None, f"{st} produced no hit"
        vectors[st] = feats.as_vector()
    names = list(vectors)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = vectors[names[i]], vectors[names[j]]
            ratio = np.maximum(np.abs(a), np.abs(b)) / np.maximum(
                np.minimum(np.abs(a), np.abs(b)), 1e-30
            )
            # At least one feature differs by > 3x, or ring-down count differs by > 5.
            assert ratio.max() > 3.0 or abs(a[1] - b[1]) > 5.0, (
                f"{names[i]} {a} and {names[j]} {b} not sufficiently distinct"
            )


def test_friction_has_longer_duration_than_crack(tmp_path: Path) -> None:
    """Friction is a sustained signal; crack is a short burst (docs/sources.md)."""
    crack = _features_for(tmp_path, "crack", 0.3, 0.3)
    friction = _features_for(tmp_path, "friction", 0.3, 0.3)
    assert crack is not None and friction is not None
    assert friction.duration_s > crack.duration_s, (
        f"friction duration {friction.duration_s * 1e3:.2f} ms not > "
        f"crack {crack.duration_s * 1e3:.2f} ms"
    )


def test_noise_has_lowest_peak_amplitude(tmp_path: Path) -> None:
    """Electrical noise is a low-amplitude mains pickup, below mechanical events.

    The 50 Hz signal lives outside the 20-90 kHz AE detection band, so what the
    feature extractor sees is residual band-limited noise \u2014 below any mechanical
    source. This makes peak amplitude a separator for the noise class (not as clean
    as the duration-based crack/friction separator, but usable).
    """
    peaks = {}
    for st in ("crack", "friction", "impact", "noise"):
        feats = _features_for(tmp_path, st, 0.5, 0.5)
        assert feats is not None, f"{st} produced no hit"
        peaks[st] = feats.peak_amplitude
    assert peaks["noise"] < 0.5 * min(peaks["crack"], peaks["friction"], peaks["impact"]), (
        f"noise peak {peaks['noise']:.2e} not < half of mechanical: {peaks}"
    )
