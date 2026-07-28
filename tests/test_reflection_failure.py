"""Sprint 1.5: the reflection-dominated failure case reports low confidence, not a
confidently wrong coordinate (ADR-006).

A source very near a plate corner (coincident with one sensor) has its direct arrival
and the first edge reflection arrive almost simultaneously at the near sensor. The
gated envelope-peak picker can catch the reflection peak instead of the direct peak,
corrupting that channel's arrival time. The multilateration residual detects this
geometric inconsistency and reports low confidence \u2014 the system says "I don't
know", not "here with high confidence".

This is the ADR-006 failure mode, deliberately demonstrated. The contrast with an
interior position makes the mechanism visible: the interior case has a large
direct-to-reflection gap on all four channels and localises with high confidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from structural_listener.cli.inject import inject_event
from structural_listener.dsp.tdoa import TdoaConfig, pick_arrival_times
from structural_listener.localize.multilateration import solve
from structural_listener.sim.config import SensorLayout


def _localise(tmp_path: Path, x: float, y: float) -> tuple[float, float, float]:
    """Return (error_m, residual_s, confidence) for an injected event at (x, y)."""
    sensors = np.array(SensorLayout().positions_m)
    path = inject_event("crack", x, y, duration_s=3.0e-3, out_dir=tmp_path)
    with np.load(path) as data:
        wf = np.asarray(data["waveforms"])
        sr = float(data["sample_rate_hz"])
        src = np.asarray(data["source_xy_m"])
    arrival_times = pick_arrival_times(wf, sr, TdoaConfig())
    result = solve(sensors, arrival_times, 1200.0)
    error = float(np.hypot(result.position_m[0] - src[0], result.position_m[1] - src[1]))
    return error, result.residual, result.confidence


def test_reflection_dominated_corner_reports_low_confidence(tmp_path: Path) -> None:
    """A source at the corner coincident with sensor 0 is reflection-dominated.

    The source at (0.05, 0.05) is essentially on top of sensor 0 (also at 0.05, 0.05).
    The direct arrival and the first edge reflection arrive within the picker's gate,
    so the picked arrival time at sensor 0 is corrupted. The residual detects this
    geometric inconsistency: it is high, and the confidence is low.

    The assertion is *not* that the error is small \u2014 it isn't (the position is
    wrong). The assertion is that the system reports low confidence, so a downstream
    consumer can reject the estimate rather than trust a confidently wrong coordinate.
    """
    error, residual, confidence = _localise(tmp_path, 0.05, 0.05)
    _ = error  # the error is large but not asserted \u2014 the point is the confidence
    # The residual is large: the picked arrival times are geometrically inconsistent.
    # 50 us is the confidence-scale threshold (residual above ~50us -> confidence < 0.5).
    assert residual > 50.0e-6, (
        f"residual {residual * 1e6:.0f} us too low \u2014 corner case not reflection-dominated; "
        f"the failure mode is not reproducing"
    )
    # Confidence is below 0.5: the system reports uncertainty, not certainty.
    assert confidence < 0.5, (
        f"confidence {confidence:.2f} >= 0.5 \u2014 the system trusts a reflection-dominated "
        f"estimate; the residual-as-confidence mitigation (ADR-006) is broken"
    )


def test_interior_position_reports_high_confidence(tmp_path: Path) -> None:
    """A central source has a large direct-to-reflection gap; confidence is high.

    This is the contrast that makes the corner failure meaningful: the same pipeline,
    on a non-degenerate geometry, reports high confidence and small error. The
    residual-as-confidence distinguishes "good geometry" from "bad geometry" \u2014
    it is not a fixed threshold, it tracks the data's geometric consistency.
    """
    error, residual, confidence = _localise(tmp_path, 0.50, 0.50)
    assert error < 0.05, f"interior error {error * 100:.1f} cm > 5 cm"
    assert residual < 20.0e-6, (
        f"interior residual {residual * 1e6:.0f} us too high \u2014 not a clean direct arrival"
    )
    assert confidence > 0.8, (
        f"interior confidence {confidence:.2f} <= 0.8 \u2014 should be high for clean geometry"
    )
