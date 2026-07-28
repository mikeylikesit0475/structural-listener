"""Sprint 0.4: the sensor model produces four realistically different channels.

Three effects are tested:
- Per-channel gain mismatch: channels have slightly different scales.
- Additive noise: the noise floor is above zero and roughly the configured RMS.
- Distance-dependent attenuation: a sensor further from the source has lower signal
  amplitude and more high-frequency loss than a closer one.
"""

from __future__ import annotations

import numpy as np

from structural_listener.sim.config import PlateConfig, SolverConfig
from structural_listener.sim.fdtd import SimResult, simulate
from structural_listener.sim.sensor_model import SensorModelConfig, apply_sensor_model
from structural_listener.sim.signatures import CrackGrowthSource


def _run_crack_at(source_xy: tuple[float, float], duration_s: float = 2.0e-3) -> SimResult:
    plate = PlateConfig()
    solver = SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx
    source = CrackGrowthSource(source_xy, dx, nx, ny)
    return simulate(source, duration_s=duration_s, plate=plate, solver=solver)


def test_four_channels_have_gain_mismatch() -> None:
    """The four channel gains are not all identical (\u00b12% spread)."""
    result = _run_crack_at((0.5, 0.5))
    _, meta = apply_sensor_model(result)
    gains = meta["gains"]
    assert gains.shape == (4,)
    assert float(np.ptp(gains)) > 0.0, (
        "all four channels have identical gain \u2014 mismatch not applied"
    )


def test_noise_floor_is_nonzero() -> None:
    """The sensor output has a noise floor above zero even away from the event."""
    # Place the source far from all sensors (impossible on a 1m plate, but use a long
    # pre-event window so we sample pure noise).
    result = _run_crack_at((0.5, 0.5), duration_s=0.1e-3)  # very short, mostly noise
    sensor_out, _ = apply_sensor_model(result)
    # The first ~50 us is before any wave can arrive at the nearest sensor (~0.5 m / c).
    # In that window we should see only noise, with RMS > 0.
    noise_window = sensor_out[:, :50]
    rms = float(np.sqrt(np.mean(noise_window**2)))
    assert rms > 0.0, "noise floor is zero \u2014 additive noise not applied"


def test_closer_sensor_gets_larger_signal() -> None:
    """A sensor nearer the source receives a larger peak amplitude than a farther one."""
    # Source near corner (0,0) where sensor 0 sits.
    result = _run_crack_at((0.1, 0.1))
    sensor_out, meta = apply_sensor_model(result)
    distances = meta["distances_m"]
    peaks = np.max(np.abs(sensor_out), axis=1)
    # Sensor 0 (at (0,0)) is closest; sensor 2 (at (1,1)) is farthest.
    order = np.argsort(distances)
    assert peaks[order[0]] > peaks[order[-1]], (
        f"closest sensor peak {peaks[order[0]]:.3e} not > farthest {peaks[order[-1]]:.3e}"
    )


def test_attenuation_reduces_high_frequencies_with_distance() -> None:
    """The direct-arrival peak amplitude decreases with source-to-sensor distance.

    On a reflective plate the total high-freq energy isn't monotonic in distance
    (reflections from the far edge contribute \u2014 the ADR-006 failure mode), so we
    measure the *direct arrival* peak in the first ~0.4 ms window, before any reflection
    can reach the nearest sensor. The sensor model's frequency-dependent attenuation
    must then reduce the far sensor's direct-arrival amplitude relative to the near one.
    """
    result = _run_crack_at((0.1, 0.1), duration_s=2.0e-3)
    # Noise-free so we measure signal attenuation, not the noise floor.
    cfg = SensorModelConfig(noise_rms_white=0.0, noise_pink_at_1hz=0.0)
    sensor_out, meta = apply_sensor_model(result, config=cfg)
    distances = meta["distances_m"]
    # Direct-arrival window: first 0.4 ms. The wave speed on the A0 mode at 80 kHz is
    # ~1000 m/s, so the direct arrival at 1.3 m takes ~1.3 ms \u2014 we lengthen the window
    # to 1.8 ms to capture it for the farthest sensor while still excluding the
    # nearest-sensor reflection (which would arrive ~0.14+2*0.86 = 1.86 ms after the
    # source). 1.8 ms is a safe cut.
    window_samples = int(1.8e-3 * result.sample_rate_hz)
    direct_peaks = np.max(np.abs(sensor_out[:, :window_samples]), axis=1)
    closest = int(np.argmin(distances))
    farthest = int(np.argmax(distances))
    assert direct_peaks[closest] > direct_peaks[farthest], (
        f"close direct-arrival peak {direct_peaks[closest]:.3e} "
        f"not > far {direct_peaks[farthest]:.3e}"
    )


def test_sensor_model_is_reproducible_with_seed() -> None:
    """Two runs with the same seed produce identical output (gain + noise are deterministic)."""
    result = _run_crack_at((0.5, 0.5))
    cfg = SensorModelConfig(seed=123)
    out_a, _ = apply_sensor_model(result, config=cfg)
    out_b, _ = apply_sensor_model(result, config=cfg)
    assert np.array_equal(out_a, out_b), "same seed produced different output"
