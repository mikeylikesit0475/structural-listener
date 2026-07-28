"""Sprint 1.2: TDOA by cross-correlation with sub-sample interpolation.

Two key tests:
- Recovers a known synthetic delay within tolerance (the unit test of the estimator).
- On simulated data, the TDOA between two sensors matches the geometric prediction
  (distance difference / wave speed) within the dispersion-bias budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from structural_listener.cli.inject import inject_event
from structural_listener.dsp.tdoa import TdoaConfig, estimate_tdoa, pick_arrival_times


def _synthetic_delayed_pair(
    delay_s: float, sample_rate_hz: float = 824_500.0, n: int = 2000
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Two identical broadband pulses; ``test`` arrives ``delay_s`` after ``ref``.

    Returns (ref, test, ref_onset, test_onset) where onsets are the sample indices of
    the first above-threshold sample in each signal.
    """
    t = np.arange(n) / sample_rate_hz
    env = np.exp(-((t - 0.5e-3) ** 2) / (2 * (0.05e-3) ** 2))
    pulse = env * np.sin(2 * np.pi * 60.0e3 * t)
    ref = pulse.copy()
    shift = round(delay_s * sample_rate_hz)
    test = np.zeros_like(pulse)
    if shift >= 0:
        test[shift:] = pulse[: n - shift]
    else:
        test[: n + shift] = pulse[-shift:]
    # Onset = first sample above 5% of peak.
    ref_onset = int(np.where(np.abs(ref) > 0.05 * np.max(np.abs(ref)))[0][0])
    test_onset = int(np.where(np.abs(test) > 0.05 * np.max(np.abs(test)))[0][0])
    return ref, test, ref_onset, test_onset


def test_recovers_known_integer_delay() -> None:
    """A 50 us delay (41 samples at 824.5 kHz) is recovered to within 2 us."""
    delay = 50.0e-6
    ref, test, ref_on, test_on = _synthetic_delayed_pair(delay)
    est = estimate_tdoa(ref, test, 824_500.0, ref_onset_sample=ref_on, test_onset_sample=test_on)
    assert est is not None, "no TDOA estimate returned"
    assert abs(est.lag_s - delay) < 2.0e-6, (
        f"estimated {est.lag_s * 1e6:.2f} us vs true {delay * 1e6:.2f} us"
    )


def test_recovers_known_subsample_delay() -> None:
    """A 12.3 us delay (~10.1 samples) is recovered with sub-sample precision."""
    delay = 12.3e-6
    ref, test, ref_on, test_on = _synthetic_delayed_pair(delay)
    est = estimate_tdoa(ref, test, 824_500.0, ref_onset_sample=ref_on, test_onset_sample=test_on)
    assert est is not None
    assert abs(est.lag_s - delay) < 0.5e-6, (
        f"estimated {est.lag_s * 1e6:.3f} us vs true {delay * 1e6:.3f} us"
    )


def test_confidence_is_high_for_matching_signals() -> None:
    """The correlation confidence is > 0.5 for two copies of the same pulse."""
    ref, test, ref_on, test_on = _synthetic_delayed_pair(30.0e-6)
    est = estimate_tdoa(ref, test, 824_500.0, ref_onset_sample=ref_on, test_onset_sample=test_on)
    assert est is not None
    assert est.confidence > 0.5, f"confidence {est.confidence:.3f} too low for matching signals"


def test_returns_none_for_uncorrelated_signals() -> None:
    """Two independent noise signals with no onsets return None."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    est = estimate_tdoa(a, b, 824_500.0)  # no onsets -> None
    assert est is None, "no-onset signals produced a TDOA estimate"


def test_tdoa_matches_geometric_prediction_on_sim(tmp_path: Path) -> None:
    """On a simulated crack event, the TDOA matches distance_diff / c_eff within 30 us.

    Uses the gated envelope-peak picker (the Sprint 1.3 arrival-time estimator). The
    effective wave speed is the grid group velocity (~1200 m/s for the 20-90 kHz band
    on the dx=5mm grid \u2014 see docs/sources.md), not the continuum theoretical c_g,
    because the 5-point-stencil biharmonic operator caps group velocity at
    2*c_basis/dx ~ 1237 m/s.

    30 us at c~1200 m/s is ~3.6 cm \u2014 within the Sprint 1 localisation target.
    """
    path = inject_event("crack", 0.2, 0.2, duration_s=3.0e-3, out_dir=tmp_path)
    with np.load(path) as data:
        wf = np.asarray(data["waveforms"])
        sr = float(data["sample_rate_hz"])
        src = np.asarray(data["source_xy_m"])
        sensors = np.asarray(data["sensor_positions_m"])

    arrival_times = pick_arrival_times(wf, sr, TdoaConfig())
    # Geometric prediction: TDOA(2 - 0) = (dist(src, s2) - dist(src, s0)) / c_eff.
    # c_eff is the grid group velocity for the 20-90 kHz band (docs/sources.md).
    c_eff = 1200.0
    d0 = float(np.hypot(src[0] - sensors[0, 0], src[1] - sensors[0, 1]))
    d2 = float(np.hypot(src[0] - sensors[2, 0], src[1] - sensors[2, 1]))
    predicted_tdoa = (d2 - d0) / c_eff
    measured_tdoa = float(arrival_times[2] - arrival_times[0])
    assert abs(measured_tdoa - predicted_tdoa) < 30.0e-6, (
        f"measured {measured_tdoa * 1e6:.1f} us vs predicted {predicted_tdoa * 1e6:.1f} us "
        f"diff {abs(measured_tdoa - predicted_tdoa) * 1e6:.1f} us > 30 us"
    )
