"""Typed acoustic-emission source signatures for the FDTD solver.

Four source types, matching the classifier labels in ARCHITECTURE.md \u00a71 and the demo
script:

- **crack-growth**: short, broadband, high-frequency burst.
- **friction**: sustained, narrowband.
- **impact**: low-frequency, high-amplitude, single transient.
- **electrical-noise**: 50 Hz + harmonics, continuous, low amplitude.

Parameters are cited in ``docs/sources.md`` (ADR-005). All four structurally conform to
``structural_listener.sim.fdtd.Source`` and share a common Gaussian spatial footprint
(see ``GaussianPulseSource`` for why the footprint is a numerical necessity, not a
physics choice).
"""

from __future__ import annotations

import numpy as np

from structural_listener.sim.fdtd import Source
from structural_listener.sim.sources import _gaussian_footprint


class _BaseSource(Source):
    """Common state for the four typed sources: position, grid footprint, time window."""

    def __init__(
        self,
        source_xy_m: tuple[float, float],
        dx: float,
        nx: int,
        ny: int,
        spatial_sigma_cells: float = 2.0,
    ) -> None:
        self._xy = source_xy_m
        self._dx = dx
        self._i = min(max(round(source_xy_m[0] / dx), 1, nx - 2), 1)
        self._j = min(max(round(source_xy_m[1] / dx), 1, ny - 2), 1)
        self._footprint = _gaussian_footprint(nx, ny, self._i, self._j, spatial_sigma_cells)

    @property
    def source_xy_m(self) -> tuple[float, float]:
        """Ground-truth source coordinates in metres (plate bottom-left origin)."""
        return self._xy

    def _footprint_force(self, amplitude: float) -> np.ndarray:
        return amplitude * self._footprint


class CrackGrowthSource(_BaseSource):
    """Short, broadband, high-frequency burst \u2014 the canonical AE crack-growth signature.

    Physically: a micro-crack advances a few microns in microseconds, releasing elastic
    energy as a broadband burst peaking in the 100\u2013300 kHz band. Our grid (dx=5mm)
    resolves up to ~100 kHz, so we centre at 80 kHz with 40 kHz bandwidth \u2014 the top of
    the resolvable band. See ``docs/sources.md`` for the resolution trade-off.
    """

    def __init__(  # noqa: D107
        self,
        source_xy_m: tuple[float, float],
        dx: float,
        nx: int,
        ny: int,
        centre_freq_hz: float = 80.0e3,
        bandwidth_hz: float = 40.0e3,
        amplitude: float = 5.0e4,
        duration_s: float = 0.3e-3,
        t0_s: float = 0.0,
    ) -> None:
        super().__init__(source_xy_m, dx, nx, ny)
        self._fc = centre_freq_hz
        self._bw = bandwidth_hz
        self._amp = amplitude
        self._dur = duration_s
        self._t0 = t0_s

    def force_field(self, t: float, w: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """Return the per-cell acceleration contribution (m/s^2) at time ``t``."""
        _ = (w, dx, dt)
        rel = t - self._t0
        if rel < -3.0 / self._bw or rel > self._dur + 3.0 / self._bw:
            return np.zeros_like(self._footprint)
        sigma_t = 1.0 / (2.0 * np.pi * self._bw)
        env = self._amp * np.exp(-((rel - self._dur * 0.5) ** 2) / (2.0 * sigma_t * sigma_t))
        carrier = np.sin(2.0 * np.pi * self._fc * rel)
        result: np.ndarray = env * carrier * self._footprint
        return result


class FrictionSource(_BaseSource):
    """Sustained, narrowband \u2014 rub/stick-slip friction produces a longer, tonal signal.

    Physically: relative sliding excites a resonance (often the sensor or a structural
    mode), giving a narrowband signal lasting milliseconds rather than microseconds.
    Centre at 30 kHz, 5 kHz bandwidth, 2 ms duration \u2014 well inside the grid band and
    clearly distinct from the crack burst in both time and frequency.
    """

    def __init__(  # noqa: D107
        self,
        source_xy_m: tuple[float, float],
        dx: float,
        nx: int,
        ny: int,
        centre_freq_hz: float = 30.0e3,
        bandwidth_hz: float = 5.0e3,
        amplitude: float = 2.0e4,
        duration_s: float = 2.0e-3,
        t0_s: float = 0.0,
    ) -> None:
        super().__init__(source_xy_m, dx, nx, ny)
        self._fc = centre_freq_hz
        self._bw = bandwidth_hz
        self._amp = amplitude
        self._dur = duration_s
        self._t0 = t0_s

    def force_field(self, t: float, w: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """Return the per-cell acceleration contribution (m/s^2) at time ``t``."""
        _ = (w, dx, dt)
        rel = t - self._t0
        if rel < -3.0 / self._bw or rel > self._dur + 3.0 / self._bw:
            return np.zeros_like(self._footprint)
        # Slow onset/offset envelope (raised cosine) \u2014 friction ramps in, doesn't slam.
        edge = 3.0 / self._bw
        env = 0.5 * (1.0 - np.cos(np.pi * np.clip(rel / edge, 0.0, 1.0)))
        if rel > self._dur - edge:
            fade = np.clip((rel - (self._dur - edge)) / edge, 0.0, 1.0)
            env *= 0.5 * (1.0 + np.cos(np.pi * fade))
        carrier = np.sin(2.0 * np.pi * self._fc * rel)
        result: np.ndarray = self._amp * env * carrier * self._footprint
        return result


class ImpactSource(_BaseSource):
    """Low-frequency, high-amplitude single transient \u2014 a mass hitting the plate.

    Physically: an impact is a single force impulse (half-sine over the contact time),
    exciting a broad low-frequency band. Centre at 8 kHz, very high amplitude, 0.5 ms
    contact. Distinct from crack by being much lower-frequency and higher-amplitude.
    """

    def __init__(  # noqa: D107
        self,
        source_xy_m: tuple[float, float],
        dx: float,
        nx: int,
        ny: int,
        centre_freq_hz: float = 8.0e3,
        amplitude: float = 1.0e5,
        duration_s: float = 0.5e-3,
        t0_s: float = 0.0,
    ) -> None:
        super().__init__(source_xy_m, dx, nx, ny)
        self._fc = centre_freq_hz
        self._amp = amplitude
        self._dur = duration_s
        self._t0 = t0_s

    def force_field(self, t: float, w: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """Return the per-cell acceleration contribution (m/s^2) at time ``t``."""
        _ = (w, dx, dt)
        rel = t - self._t0
        if rel < 0.0 or rel > self._dur:
            return np.zeros_like(self._footprint)
        # Half-sine force pulse over the contact duration.
        env = self._amp * np.sin(np.pi * rel / self._dur)
        carrier = np.sin(2.0 * np.pi * self._fc * rel)
        result: np.ndarray = env * carrier * self._footprint
        return result


class ElectricalNoiseSource(_BaseSource):
    """50 Hz + harmonics continuous electrical interference \u2014 not a mechanical event.

    Physically: mains pickup couples into the sensor leads, producing a tonal signal at
    the mains frequency and its harmonics. This is a *detection* class, not a localisation
    target \u2014 it has no meaningful source position, but we keep a nominal coordinate for
    pipeline uniformity. Amplitude is low compared to real events.
    """

    def __init__(  # noqa: D107
        self,
        source_xy_m: tuple[float, float],
        dx: float,
        nx: int,
        ny: int,
        mains_freq_hz: float = 50.0,
        n_harmonics: int = 3,
        amplitude: float = 200.0,
        t0_s: float = 0.0,
    ) -> None:
        super().__init__(source_xy_m, dx, nx, ny)
        self._f0 = mains_freq_hz
        self._n_harm = n_harmonics
        self._amp = amplitude
        self._t0 = t0_s

    def force_field(self, t: float, w: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """Return the per-cell acceleration contribution (m/s^2) at time ``t``."""
        _ = (w, dx, dt)
        rel = t - self._t0
        signal = 0.0
        for k in range(1, self._n_harm + 1):
            # Harmonics decay as 1/k \u2014 a guess, but the standard first-order model.
            signal += np.sin(2.0 * np.pi * self._f0 * k * rel) / k
        # Normalise so the fundamental has unit amplitude, then scale.
        signal *= self._amp
        result: np.ndarray = signal * self._footprint
        return result


# Registry for CLI / dataset generation. Names match the `make <type>` targets.
SOURCE_REGISTRY: dict[str, type] = {
    "crack": CrackGrowthSource,
    "friction": FrictionSource,
    "impact": ImpactSource,
    "noise": ElectricalNoiseSource,
}
