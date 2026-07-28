"""Physical and numerical constants for the FDTD thin-plate solver.

Every value here is either cited in ``docs/sources.md`` or flagged as a guess, per ADR-005.
The short citation key in each docstring maps to a full reference in that file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlateConfig:
    """1 m x 1 m aluminium 6061-T6 plate, thin-plate (Kirchhoff) approximation.

    The plate carries the fundamental A0 (flexural) Lamb mode; S0 is out of scope for
    Sprint 0 (see ARCHITECTURE.md \u00a73 and the parking lot).
    """

    length: float = 1.0
    width: float = 1.0
    thickness: float = 2.0e-3
    youngs_modulus: float = 69.0e9
    poisson: float = 0.33
    density: float = 2700.0

    @property
    def flexural_rigidity(self) -> float:
        """Bending stiffness D = E h^3 / (12 (1 - nu^2)), in N\u00b7m."""
        return self.youngs_modulus * self.thickness**3 / (12.0 * (1.0 - self.poisson**2))

    @property
    def mass_per_area(self) -> float:
        """Mass per unit area (rho h), in kg/m^2."""
        return self.density * self.thickness


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Numerical grid and time-stepping parameters.

    Default ``dx = 5 mm`` is chosen against the Sprint 0.2 wall-clock budget (< 5 s / event):
    on the dev box (Ryzen AI Max+ 395, NumPy CPU) a 5 ms simulation at N=200 takes ~370 ms.
    The resolution cost: ~2 grid points per wavelength at 100 kHz on the A0 mode
    (wavelength ~10.5 mm at 100 kHz) \u2014 marginal at the top of the band, adequate below
    ~50 kHz where the source signatures concentrate. See ``docs/sources.md`` for the
    wavelength calculation. Coarsening further is the documented mitigation if a later
    sprint needs more headroom (CLAUDE.md).
    """

    dx: float = 5.0e-3
    cfl_safety: float = 0.6
    boundary: str = "clamped"

    @property
    def nx(self) -> int:
        """Grid count along the plate length."""
        return round(PlateConfig().length / self.dx)

    @property
    def ny(self) -> int:
        """Grid count along the plate width."""
        return round(PlateConfig().width / self.dx)


@dataclass(frozen=True, slots=True)
class SensorLayout:
    """Four piezo sensors near the plate corners, indexed counter-clockwise.

    Sensors are placed a few cells *inward* from the corners (2 cm on a 1 m plate) to
    avoid numerical artifacts from the clamped-boundary zeroing at the grid edge.
    Corner placement maximises the baseline aperture for multilateration (Sprint 1) and
    matches the demo in ARCHITECTURE.md \u00a71. Coordinates are in metres, origin at the
    bottom-left corner of the plate.

    Attributes:
        inset_m: distance from each edge to the sensor position, in metres.

    """

    inset_m: float = 0.05  # 5 cm from each edge; avoids boundary numerical artifacts

    def positions_for_plate(self, plate: PlateConfig) -> tuple[tuple[float, float], ...]:
        """Return sensor positions scaled to the given plate dimensions."""
        x0 = self.inset_m
        x1 = plate.length - self.inset_m
        y0 = self.inset_m
        y1 = plate.width - self.inset_m
        return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))

    @property
    def positions_m(self) -> tuple[tuple[float, float], ...]:
        """Sensor positions for the default 1 m plate (backwards compatibility)."""
        return self.positions_for_plate(PlateConfig())

    def as_grid_indices(self, dx: float, nx: int, ny: int) -> tuple[tuple[int, int], ...]:
        """Return (i, j) grid indices for each sensor, clamped to the interior grid.

        Sensors nominally sit on the boundary; we step one cell inward so clamped-boundary
        reflections don't alias the sensor sample.
        """
        lo = 1
        hi_x = max(nx - 2, lo)
        hi_y = max(ny - 2, lo)
        return tuple(
            (
                min(max(round(x / dx), lo), hi_x),
                min(max(round(y / dx), lo), hi_y),
            )
            for x, y in self.positions_m
        )
