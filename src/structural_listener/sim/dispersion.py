"""Dispersion measurement for the A0 flexural mode.

The Kirchhoff thin-plate equation already models the A0 mode's dispersion
(``c_g(f) = (4*pi^2*D/(rho h))^{1/4} * sqrt(f)``). What this module does is *measure* the
consequence for time-of-arrival estimation: a broadband pulse's apparent arrival time
depends on the frequency band used to detect it, because higher frequencies travel
faster. This is the systematic error Sprint 1's TDOA pipeline must account for, and
ADR-003's "arrival time is estimated at a fixed band" is the mitigation.

The measurement: inject a broadband pulse at a known position, record the arrival at a
sensor a known distance away, and compare the group delay in a low band vs a high band.
The difference is the dispersion-induced TDOA bias \u2014 the number that goes in
``docs/dispersion-bias.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from structural_listener.sim.config import PlateConfig, SensorLayout, SolverConfig
from structural_listener.sim.fdtd import simulate
from structural_listener.sim.signatures import CrackGrowthSource


@dataclass(frozen=True, slots=True)
class DispersionMeasurement:
    """Result of a single dispersion-bias measurement.

    Attributes:
        distance_m: source-to-sensor distance.
        low_band_hz: (centre, half-width) of the low detection band.
        high_band_hz: (centre, half-width) of the high detection band.
        arrival_low_s: estimated arrival time in the low band.
        arrival_high_s: estimated arrival time in the high band.
        bias_s: arrival_high - arrival_low (negative: high band arrives earlier).
        c_g_low_m_per_s: implied group velocity in the low band.
        c_g_high_m_per_s: implied group velocity in the high band.

    """

    distance_m: float
    low_band_hz: tuple[float, float]
    high_band_hz: tuple[float, float]
    arrival_low_s: float
    arrival_high_s: float
    bias_s: float
    c_g_low_m_per_s: float
    c_g_high_m_per_s: float


def _bandpass_arrival_time(
    wave: np.ndarray, sample_rate_hz: float, centre_hz: float, half_width_hz: float
) -> float:
    """Estimate the arrival time of the band-limited energy envelope.

    Bandpass around (centre, half-width) via FFT zeroing, take the magnitude envelope
    (analytic signal), and return the time of the first sample exceeding 20% of the
    envelope peak. This is the same detection rule Sprint 1's hit detector will use.
    """
    n = wave.shape[0]
    spec = np.fft.rfft(wave * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    mask = (freqs >= centre_hz - half_width_hz) & (freqs <= centre_hz + half_width_hz)
    spec_band = spec * mask
    band = np.fft.irfft(spec_band, n=n)
    # Analytic envelope.
    analytic = np.fft.irfft(np.fft.rfft(band) * 2.0 * mask, n=n)
    envelope = np.abs(analytic)
    peak = float(np.max(envelope))
    if peak <= 0.0:
        return float(n) / sample_rate_hz
    above = np.where(envelope >= 0.2 * peak)[0]
    return float(above[0]) / sample_rate_hz if above.size else float(n) / sample_rate_hz


def measure_dispersion_bias(
    source_xy_m: tuple[float, float] = (0.15, 0.15),
    sensor_idx: int = 1,
    low_band_hz: tuple[float, float] = (20.0e3, 10.0e3),
    high_band_hz: tuple[float, float] = (70.0e3, 10.0e3),
    duration_s: float = 3.0e-3,
    plate: PlateConfig | None = None,
    solver: SolverConfig | None = None,
) -> DispersionMeasurement:
    """Measure the dispersion-induced arrival-time bias between two frequency bands.

    Args:
        source_xy_m: source position; should be near one corner so the chosen sensor
            receives a clean direct arrival before reflections.
        sensor_idx: which of the four corner sensors to measure at (0..3).
        low_band_hz: (centre, half-width) for the low detection band.
        high_band_hz: (centre, half-width) for the high detection band.
        duration_s: simulation duration; long enough for the direct arrival at the
            chosen sensor but short enough to exclude reflections.
        plate: plate config. If None, uses ``PlateConfig()`` defaults.
        solver: solver config. If None, uses ``SolverConfig()`` defaults.

    """
    plate = plate or PlateConfig()
    solver = solver or SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx

    source = CrackGrowthSource(source_xy_m, dx, nx, ny)
    result = simulate(source, duration_s=duration_s, plate=plate, solver=solver)
    sensor_pos = SensorLayout().positions_for_plate(plate)[sensor_idx]
    distance = float(np.hypot(sensor_pos[0] - source_xy_m[0], sensor_pos[1] - source_xy_m[1]))

    wave = result.waveforms[sensor_idx]
    t_low = _bandpass_arrival_time(wave, result.sample_rate_hz, low_band_hz[0], low_band_hz[1])
    t_high = _bandpass_arrival_time(wave, result.sample_rate_hz, high_band_hz[0], high_band_hz[1])

    c_g_low = distance / t_low if t_low > 0 else float("inf")
    c_g_high = distance / t_high if t_high > 0 else float("inf")

    return DispersionMeasurement(
        distance_m=distance,
        low_band_hz=low_band_hz,
        high_band_hz=high_band_hz,
        arrival_low_s=t_low,
        arrival_high_s=t_high,
        bias_s=t_high - t_low,
        c_g_low_m_per_s=c_g_low,
        c_g_high_m_per_s=c_g_high,
    )


def theoretical_group_velocity(freq_hz: float, plate: PlateConfig | None = None) -> float:
    """Theoretical A0 group velocity: c_g(f) = 2 * (4*pi^2*D/(rho h))^{1/4} * sqrt(f).

    For the Kirchhoff dispersion relation omega = sqrt(D/(rho h)) * k^2, the group
    velocity d omega/d k is exactly twice the phase velocity omega/k.
    """
    plate = plate or PlateConfig()
    d = plate.flexural_rigidity
    rhoh = plate.mass_per_area
    return float(2.0 * (4.0 * np.pi**2 * d / rhoh) ** 0.25 * np.sqrt(freq_hz))
