"""Sprint 1.3: Gauss-Newton multilateration recovers a known source to < 5 cm.

The CI-asserted test: inject events at 100 positions across the plate, run the full
pipeline (gated envelope-peak picker -> multilateration), and assert the median
localisation error is < 5 cm. This is the Sprint 1 definition of done (BACKLOG.md).

Also tests the solver in isolation with synthetic arrival times (no noise, no
dispersion) to verify the maths, and the residual-as-confidence behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from structural_listener.cli.inject import inject_event
from structural_listener.dsp.tdoa import TdoaConfig, pick_arrival_times
from structural_listener.localize.multilateration import (
    solve,
)
from structural_listener.sim.config import SensorLayout

# Effective wave speed for the gated envelope-peak picker on the 20-90 kHz band,
# on the dx=5 mm grid. The 5-point-Laplacian biharmonic operator caps the group
# velocity at 2 * c_basis / dx ~ 1237 m/s, and the band sits above ~20 kHz where the
# grid dispersion is near its asymptote, so the measured c_eff is ~1200 m/s. This is
# a grid property, not a calibration hack \u2014 see docs/sources.md.
_C_EFF_M_PER_S = 1200.0


def _synthetic_arrival_times(
    source_xy: tuple[float, float],
    sensor_positions: np.ndarray,
    wave_speed: float,
) -> np.ndarray:
    """Perfect (noise-free) arrival times: distance / wave_speed for each sensor."""
    times = np.empty(sensor_positions.shape[0])
    for i in range(sensor_positions.shape[0]):
        d = float(
            np.hypot(source_xy[0] - sensor_positions[i, 0], source_xy[1] - sensor_positions[i, 1])
        )
        times[i] = d / wave_speed
    return times


def test_recovers_known_source_with_perfect_arrival_times() -> None:
    """With noise-free arrival times, the solver recovers the source to < 1 mm."""
    sensors = np.array(SensorLayout().positions_m)
    source = (0.37, 0.62)
    c = 1000.0
    times = _synthetic_arrival_times(source, sensors, c)
    result = solve(sensors, times, c)
    error = float(np.hypot(result.position_m[0] - source[0], result.position_m[1] - source[1]))
    # 1 mm tolerance: the solver converges to machine precision with perfect data.
    assert error < 1.0e-3, f"error {error * 1e3:.3f} mm > 1 mm with perfect arrival times"
    assert result.converged
    # Residual < 1 us: the convergence tolerance (0.1 mm) gives ~1e-7 s residual.
    assert result.residual < 1.0e-6


def test_fails_gracefully_with_fewer_than_three_sensors() -> None:
    """The solver raises ValueError with < 3 valid arrival times."""
    sensors = np.array(SensorLayout().positions_m)
    times = np.array([0.0, np.nan, np.nan, 0.001])
    with pytest.raises(ValueError, match=">= 3 sensors"):
        solve(sensors, times, 1000.0)


def test_residual_is_high_for_inconsistent_arrival_times() -> None:
    """Geometrically inconsistent arrival times produce a high residual (ADR-006)."""
    sensors = np.array(SensorLayout().positions_m)
    # Arrival times that don't correspond to any physical source position.
    times = np.array([0.0, 0.001, 0.0005, 0.002])
    result = solve(sensors, times, 1000.0)
    # The residual should be non-trivial (the times are inconsistent).
    assert result.residual > 10.0e-6, (
        f"residual {result.residual * 1e6:.1f} us too low for inconsistent arrivals"
    )
    assert result.confidence < 0.9, "inconsistent arrivals gave high confidence"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_recovers_source_from_simulated_pipeline(tmp_path: Path, seed: int) -> None:
    """End-to-end: inject -> gated envelope-peak TDOA -> multilateration -> < 12 cm.

    Each individual position must localise within 12 cm. The 100-position grid test
    below asserts the < 5 cm *median*; here we assert a per-position ceiling that
    accommodates near-edge positions where the direct-to-reflection gap is smallest.
    """
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(0.2, 0.8))
    y = float(rng.uniform(0.2, 0.8))

    path = inject_event("crack", x, y, duration_s=3.0e-3, out_dir=tmp_path)
    with np.load(path) as data:
        wf = np.asarray(data["waveforms"])
        sr = float(data["sample_rate_hz"])
        src = np.asarray(data["source_xy_m"])
        sensors = np.asarray(data["sensor_positions_m"])

    arrival_times = pick_arrival_times(wf, sr, TdoaConfig())
    result = solve(sensors, arrival_times, _C_EFF_M_PER_S)
    error = float(np.hypot(result.position_m[0] - src[0], result.position_m[1] - src[1]))
    assert result.converged, "solver did not converge"
    assert error < 0.12, (
        f"source ({x:.2f},{y:.2f}): estimated {result.position_m}, "
        f"error {error * 100:.1f} cm > 12 cm, residual {result.residual * 1e6:.0f} us"
    )


def test_median_error_under_5cm_over_100_positions(tmp_path: Path) -> None:
    """The Sprint 1 CI assertion: median localisation error < 5 cm over 100 positions.

    A 10x10 grid of source positions across the plate interior (0.15-0.85 m). The
    median is asserted (not the max) because near-edge positions have higher error
    due to reflections (ADR-006) \u2014 the max is studied in Sprint 1.4.

    If this fails, the TDOA picker is catching reflections instead of direct arrivals
    \u2014 the fix is in the picker, not the solver. The solver maths is verified
    separately by ``test_recovers_known_source_with_perfect_arrival_times``.
    """
    sensors = np.array(SensorLayout().positions_m)
    errors = []
    grid = np.linspace(0.15, 0.85, 10)

    for x in grid:
        for y in grid:
            path = inject_event("crack", float(x), float(y), duration_s=3.0e-3, out_dir=tmp_path)
            with np.load(path) as data:
                wf = np.asarray(data["waveforms"])
                sr = float(data["sample_rate_hz"])
                src = np.asarray(data["source_xy_m"])

            arrival_times = pick_arrival_times(wf, sr, TdoaConfig())
            try:
                result = solve(sensors, arrival_times, _C_EFF_M_PER_S)
            except ValueError:
                continue
            error = float(np.hypot(result.position_m[0] - src[0], result.position_m[1] - src[1]))
            errors.append(error)

    assert len(errors) >= 80, f"only {len(errors)}/100 positions produced a localisation"
    median_error = float(np.median(errors))
    # 5 cm median: the Sprint 1 target (BACKLOG.md). Not a bare approx \u2014 the gated
    # envelope-peak picker on a 20-90 kHz band with c_eff=1200 m/s achieves ~3 cm
    # median after the source-injection bug fix.
    assert median_error < 0.05, (
        f"median error {median_error * 100:.1f} cm > 5 cm over {len(errors)} positions; "
        f"mean {float(np.mean(errors)) * 100:.1f} cm, "
        f"p90 {float(np.percentile(errors, 90)) * 100:.1f} cm"
    )
