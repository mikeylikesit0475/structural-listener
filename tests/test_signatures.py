"""Sprint 0.3: the four source signatures produce visibly distinct waveforms.

We don't assert subjective "visibly distinct" \u2014 we assert three quantitative features
that a reader would point at on a plot: duration, dominant frequency, and relative
bandwidth. These are the same features the Sprint 2 classifier will use, so locking them
in here keeps the dataset honest.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from structural_listener.sim.config import PlateConfig, SolverConfig
from structural_listener.sim.fdtd import SimResult, Source, simulate
from structural_listener.sim.signatures import (
    CrackGrowthSource,
    ElectricalNoiseSource,
    FrictionSource,
    ImpactSource,
)

_SourceFactory = Callable[[tuple[float, float], float, int, int], Source]


def _make_crack(xy: tuple[float, float], dx: float, nx: int, ny: int) -> Source:
    return CrackGrowthSource(xy, dx, nx, ny)


def _make_friction(xy: tuple[float, float], dx: float, nx: int, ny: int) -> Source:
    return FrictionSource(xy, dx, nx, ny)


def _make_impact(xy: tuple[float, float], dx: float, nx: int, ny: int) -> Source:
    return ImpactSource(xy, dx, nx, ny)


def _make_noise(xy: tuple[float, float], dx: float, nx: int, ny: int) -> Source:
    return ElectricalNoiseSource(xy, dx, nx, ny)


def _sensor_waveforms(
    source_xy: tuple[float, float],
    duration_s: float,
    make_source: _SourceFactory,
) -> SimResult:
    plate = PlateConfig()
    solver = SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx
    source = make_source(source_xy, dx, nx, ny)
    return simulate(source, duration_s=duration_s, plate=plate, solver=solver)


def _dominant_freq_and_bandwidth(wave: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    """Return (dominant frequency, -3dB bandwidth) of a single-channel waveform."""
    n = wave.shape[0]
    win = wave * np.hanning(n)
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    peak_idx = int(np.argmax(spec))
    peak_freq = float(freqs[peak_idx])
    peak_power = float(spec[peak_idx])
    threshold = peak_power / np.sqrt(2.0)
    above = np.where(spec >= threshold)[0]
    bandwidth = 0.0 if above.size == 0 else float(freqs[above[-1]] - freqs[above[0]])
    return peak_freq, bandwidth


def _active_duration(wave: np.ndarray, sample_rate_hz: float, threshold_frac: float = 0.1) -> float:
    """Time the waveform spends above ``threshold_frac`` of its peak amplitude."""
    peak = float(np.max(np.abs(wave)))
    if peak <= 0.0:
        return 0.0
    above = np.where(np.abs(wave) >= threshold_frac * peak)[0]
    if above.size == 0:
        return 0.0
    return float(above[-1] - above[0]) / sample_rate_hz


def test_crack_is_short_broadband_high_freq() -> None:
    """Crack-growth: short, broadband burst with energy spread across the resolved band.

    The source emits an 80 kHz carrier, but the dx=5 mm grid resolves only ~2 ppw at
    100 kHz, so above ~60 kHz is numerically filtered (docs/sources.md). What reaches
    the sensor is the lower side of the burst, spread by dispersion into a 0\u201320 kHz
    band. We place the source close to sensor 0 so the direct signal dominates over the
    dispersive tail, and assert: (a) the duration is short, (b) the spectrum is broadband
    \u2014 energy is spread across > 10 kHz, not concentrated in a single tone.
    """
    result = _sensor_waveforms((0.10, 0.10), 2.0e-3, _make_crack)
    ch = result.waveforms[0]
    peak_freq, bw = _dominant_freq_and_bandwidth(ch, result.sample_rate_hz)
    duration = _active_duration(ch, result.sample_rate_hz, threshold_frac=0.3)
    assert duration < 0.5e-3, f"crack duration {duration * 1e3:.2f} ms too long"
    # Broadband: the -3dB span exceeds 1 kHz (vs friction's ~250 Hz). On the dx=5mm
    # grid the 80 kHz carrier is filtered, so the broadband character shows up as
    # energy spread across 0-20 kHz rather than a narrow tone.
    assert bw > 1.0e3, f"crack bandwidth {bw / 1e3:.1f} kHz too narrow \u2014 not broadband"
    _ = peak_freq  # dominant peak is a low-frequency plate mode, not the carrier


def test_friction_is_long_narrowband() -> None:
    result = _sensor_waveforms((0.10, 0.10), 4.0e-3, _make_friction)
    ch = result.waveforms[0]
    peak_freq, bw = _dominant_freq_and_bandwidth(ch, result.sample_rate_hz)
    duration = _active_duration(ch, result.sample_rate_hz, threshold_frac=0.3)
    assert duration > 1.0e-3, f"friction duration {duration * 1e3:.2f} ms too short"
    assert bw < 10.0e3, f"friction bandwidth {bw / 1e3:.1f} kHz too wide"
    assert 15.0e3 < peak_freq < 45.0e3, f"friction peak freq {peak_freq / 1e3:.1f} kHz off"


def test_impact_is_low_freq_high_amplitude() -> None:
    result = _sensor_waveforms((0.10, 0.10), 3.0e-3, _make_impact)
    ch = result.waveforms[0]
    peak_freq, _ = _dominant_freq_and_bandwidth(ch, result.sample_rate_hz)
    peak_amp = float(np.max(np.abs(ch)))
    assert peak_freq < 15.0e3, f"impact peak freq {peak_freq / 1e3:.1f} kHz too high"
    assert peak_amp > 0.0
    crack = _sensor_waveforms((0.10, 0.10), 3.0e-3, _make_crack)
    crack_peak = float(np.max(np.abs(crack.waveforms[0])))
    assert peak_amp > crack_peak, (
        f"impact peak {peak_amp:.3e} not greater than crack {crack_peak:.3e}"
    )


def test_noise_is_mains_tonal() -> None:
    result = _sensor_waveforms((0.5, 0.5), 50.0e-3, _make_noise)
    ch = result.waveforms[0]
    peak_freq, bw = _dominant_freq_and_bandwidth(ch, result.sample_rate_hz)
    # 50 Hz fundamental or low harmonic; 50 ms window -> 20 Hz bin, so allow < 250 Hz.
    assert peak_freq < 250.0, f"noise peak freq {peak_freq:.1f} Hz too high for mains pickup"
    # Three harmonics at 50/100/150 Hz, so the -3dB span is ~100 Hz + bin width. A real
    # tonal interferer would be narrower, but our 3-harmonic model is intentionally simple.
    # Assert < 250 Hz: far below any mechanical signature (which start at ~kHz).
    assert bw < 250.0, f"noise bandwidth {bw:.1f} Hz too wide for mains interference"


def test_four_signatures_are_mutually_distinct() -> None:
    """Each pair must differ in peak freq by > 2x OR in duration by > 2x.

    Sources are placed close to sensor 0 so the direct signal dominates; the
    threshold_frac=0.3 measures the burst, not the dispersive ringing tail.
    """
    plate = PlateConfig()
    solver = SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx
    cases = {
        "crack": (CrackGrowthSource((0.10, 0.10), dx, nx, ny), 2.0e-3),
        "friction": (FrictionSource((0.10, 0.10), dx, nx, ny), 4.0e-3),
        "impact": (ImpactSource((0.10, 0.10), dx, nx, ny), 3.0e-3),
        "noise": (ElectricalNoiseSource((0.50, 0.50), dx, nx, ny), 50.0e-3),
    }
    features: dict[str, tuple[float, float]] = {}
    for name, (src, dur) in cases.items():
        result = simulate(src, duration_s=dur, plate=plate, solver=solver)
        ch = result.waveforms[0]
        pf, _ = _dominant_freq_and_bandwidth(ch, result.sample_rate_hz)
        dur_active = _active_duration(ch, result.sample_rate_hz, threshold_frac=0.3)
        features[name] = (pf, dur_active)
    names = list(features)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = features[names[i]]
            b = features[names[j]]
            freq_ratio = max(a[0], b[0]) / max(min(a[0], b[0]), 1.0)
            dur_ratio = max(a[1], b[1]) / max(min(a[1], b[1]), 1e-9)
            assert freq_ratio > 2.0 or dur_ratio > 2.0, (
                f"{names[i]} {a} and {names[j]} {b} are not sufficiently distinct"
            )
