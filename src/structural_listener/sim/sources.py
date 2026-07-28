"""Source signatures for the FDTD solver.

Sprint 0.3 fills in the four typed AE signatures (crack / friction / impact / noise) with
literature-cited parameters. This module currently provides a minimal point-source burst
used by the Sprint 0.2 energy-conservation test and grid-sizing checks.
"""

from __future__ import annotations

import numpy as np


def _gaussian_footprint(nx: int, ny: int, i0: int, j0: int, sigma_cells: float) -> np.ndarray:
    """Build a 2D Gaussian centred at (i0, j0) with FWHM ~ 2.355 * sigma_cells, on the grid.

    Spreading the source over a few cells keeps the excited wavenumber content inside the
    band the discrete biharmonic operator can stably integrate. A true delta-function
    source excites near-Nyquist modes that destabilise the leapfrog scheme even within
    CFL \u2014 this is a numerical necessity, not a physics choice.
    """
    ys, xs = np.mgrid[0:nx, 0:ny]
    g: np.ndarray = np.exp(-((xs - j0) ** 2 + (ys - i0) ** 2) / (2.0 * sigma_cells * sigma_cells))
    g /= float(np.sum(g))
    return g


class GaussianPulseSource:
    """A Gaussian-spatial-footprint, Gaussian-windowed sinusoid point source.

    Sprint 0.2 stand-in. Real crack-growth signatures land in Sprint 0.3 with cited
    centre frequencies and bandwidths; this gives the solver something to radiate while
    the energy-conservation test is written. Structurally conforms to
    ``structural_listener.sim.fdtd.Source``.
    """

    def __init__(  # noqa: D107
        self,
        source_xy_m: tuple[float, float],
        centre_freq_hz: float = 60.0e3,
        bandwidth_hz: float = 20.0e3,
        amplitude: float = 1.0e3,
        t0_s: float = 0.0,
        duration_s: float = 0.5e-3,
        dx: float = 5.0e-3,
        nx: int = 200,
        ny: int = 200,
        spatial_sigma_cells: float = 2.0,
    ) -> None:
        self._xy = source_xy_m
        self._fc = centre_freq_hz
        self._bw = bandwidth_hz
        self._amp = amplitude
        self._t0 = t0_s
        self._dur = duration_s
        self._dx = dx
        self._i = min(max(round(source_xy_m[0] / dx), 1), nx - 2)
        self._j = min(max(round(source_xy_m[1] / dx), 1), ny - 2)
        self._footprint = _gaussian_footprint(nx, ny, self._i, self._j, spatial_sigma_cells)

    @property
    def source_xy_m(self) -> tuple[float, float]:
        """Ground-truth source coordinates in metres (plate bottom-left origin)."""
        return self._xy

    def force_field(self, t: float, w: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """Return the per-cell acceleration contribution (m/s^2) at time ``t``."""
        _ = (w, dx, dt)  # forcing is independent of the current field state
        rel = t - self._t0
        if rel < -3.0 / self._bw or rel > self._dur + 3.0 / self._bw:
            return np.zeros_like(self._footprint)
        sigma_t = 1.0 / (2.0 * np.pi * self._bw)
        env = self._amp * np.exp(-((rel - self._dur * 0.5) ** 2) / (2.0 * sigma_t * sigma_t))
        carrier = np.sin(2.0 * np.pi * self._fc * rel)
        result: np.ndarray = env * carrier * self._footprint
        return result
