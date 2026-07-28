"""Sprint 0.5: dispersion \u2014 the A0 mode's group velocity is frequency-dependent.

The Kirchhoff thin-plate equation already models A0 dispersion. These tests measure the
*consequence*: a broadband pulse's apparent arrival time depends on the detection band.
The bias is recorded in ``docs/dispersion-bias.md`` and bounds the systematic error
Sprint 1's TDOA pipeline must account for.
"""

from __future__ import annotations

from structural_listener.sim.config import PlateConfig
from structural_listener.sim.dispersion import (
    measure_dispersion_bias,
    theoretical_group_velocity,
)


def test_high_band_arrives_earlier_than_low_band() -> None:
    """A0 dispersion: higher frequencies have higher group velocity, so arrive first.

    This is the central fact that makes TDOA *hard* and worth demonstrating
    (ARCHITECTURE.md \u00a73). If this test passes, the solver is producing real dispersion.
    """
    m = measure_dispersion_bias(
        source_xy_m=(0.05, 0.05),
        sensor_idx=1,  # sensor at (1,0), distance ~0.95 m
        low_band_hz=(20.0e3, 10.0e3),
        high_band_hz=(70.0e3, 10.0e3),
        duration_s=3.0e-3,
    )
    assert m.bias_s < 0.0, (
        f"high band arrived {m.bias_s * 1e6:.1f} us later than low \u2014 dispersion not modelled"
    )


def test_measured_group_velocity_matches_theory_within_tolerance() -> None:
    """The implied group velocity in each band matches the A0 theory within 50%.

    50% is a wide tolerance, deliberately. The bandpass arrival estimator catches the
    envelope *onset* (first sample > 20% peak), which for a broadband pulse includes
    slower-frequency components leaking through the bandpass skirts \u2014 so the measured
    arrival is systematically later than a pure tone would predict, and the implied c_g
    is lower than theory. The point of this test is to confirm the *order of magnitude
    and trend* match the A0 prediction (c_g proportional to sqrt(f)), not to nail the
    coefficient. Tighter matching would need a narrower source spectrum and longer path
    \u2014 both cost runtime and are out of scope for the timebox (ARCHITECTURE.md \u00a78).
    """
    plate = PlateConfig()
    m = measure_dispersion_bias(
        source_xy_m=(0.05, 0.05),
        sensor_idx=1,
        low_band_hz=(20.0e3, 5.0e3),
        high_band_hz=(60.0e3, 5.0e3),
        duration_s=3.0e-3,
        plate=plate,
    )
    c_theory_low = theoretical_group_velocity(m.low_band_hz[0], plate)
    c_theory_high = theoretical_group_velocity(m.high_band_hz[0], plate)
    rel_low = abs(m.c_g_low_m_per_s - c_theory_low) / c_theory_low
    rel_high = abs(m.c_g_high_m_per_s - c_theory_high) / c_theory_high
    # 50% tolerance: see docstring. The systematic underestimate is documented, not a bug.
    assert rel_low < 0.50, (
        f"low-band c_g {m.c_g_low_m_per_s:.0f} vs theory {c_theory_low:.0f} m/s ({rel_low:.2%})"
    )
    assert rel_high < 0.50, (
        f"high-band c_g {m.c_g_high_m_per_s:.0f} vs theory {c_theory_high:.0f} m/s ({rel_high:.2%})"
    )
    # The trend must be correct: high band is faster than low band.
    assert m.c_g_high_m_per_s > m.c_g_low_m_per_s, (
        "high-band c_g not > low-band c_g \u2014 dispersion trend is wrong"
    )


def test_dispersion_bias_is_recorded() -> None:
    """The measured bias is non-trivial (> 5 us) \u2014 large enough to matter for TDOA.

    A 5 us bias at c~1000 m/s is a 5 mm position error. Sprint 1's < 5 cm target leaves
    10x headroom, so this bounds the systematic and confirms it must be handled by
    fixed-band detection (ADR-003).
    """
    m = measure_dispersion_bias(
        source_xy_m=(0.05, 0.05),
        sensor_idx=1,
        low_band_hz=(20.0e3, 10.0e3),
        high_band_hz=(70.0e3, 10.0e3),
        duration_s=3.0e-3,
    )
    assert abs(m.bias_s) > 5.0e-6, (
        f"dispersion bias {abs(m.bias_s) * 1e6:.1f} us < 5 us \u2014 too small to be the A0 effect"
    )
