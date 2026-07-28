"""Localisation error distribution over a grid of source positions.

Sprint 1.4. The study: inject crack events at a grid of positions across the plate,
run the full pipeline (gated envelope-peak picker -> Gauss-Newton multilateration),
and record the per-position error and the multilateration residual. The output is a
heat map of error vs. source position, committed to ``docs/figures/error-heatmap.png``.

The heat map makes the ADR-006 failure mode visible: near-edge positions have higher
error because the direct-to-first-reflection time gap is smallest there, so the
gated picker is more likely to catch the reflection. The interior positions achieve
the < 5 cm target; the edges degrade. The residual-as-confidence reports this \u2014
the system says "low confidence" rather than a confidently wrong coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from structural_listener.cli.inject import inject_event
from structural_listener.dsp.tdoa import TdoaConfig, pick_arrival_times
from structural_listener.localize.multilateration import solve
from structural_listener.sim.config import SensorLayout

# Effective wave speed for the gated envelope-peak picker on the 20-90 kHz band on the
# dx=5mm grid. Cited in docs/sources.md as the grid group-velocity cap.
C_EFF_M_PER_S = 1200.0


@dataclass(frozen=True, slots=True)
class GridPosition:
    """One grid point's localisation outcome.

    Attributes:
        source_xy_m: the true source position (metres).
        estimated_xy_m: the multilateration estimate (metres), or NaN if the solver
            could not run (fewer than 3 sensors produced an arrival pick).
        error_m: |estimated - true| in metres, or NaN.
        residual_s: the multilateration residual (seconds), or NaN.
        confidence: the solver's confidence score in [0, 1], or NaN.

    """

    source_xy_m: tuple[float, float]
    estimated_xy_m: tuple[float, float]
    error_m: float
    residual_s: float
    confidence: float


@dataclass(frozen=True, slots=True)
class GridStudy:
    """The full grid study result.

    Attributes:
        positions: list of per-position outcomes, row-major over the grid.
        grid_x_m: the grid x coordinates (metres).
        grid_y_m: the grid y coordinates (metres).
        n_localised: number of positions that produced a localisation.
        median_error_m: median error over localised positions.
        mean_error_m: mean error over localised positions.
        p90_error_m: 90th percentile error over localised positions.
        max_error_m: max error over localised positions.

    """

    positions: list[GridPosition]
    grid_x_m: np.ndarray
    grid_y_m: np.ndarray
    n_localised: int
    median_error_m: float
    mean_error_m: float
    p90_error_m: float
    max_error_m: float


def run_grid_study(
    out_dir: Path,
    grid_step: int = 10,
    grid_extent: tuple[float, float] = (0.15, 0.85),
    duration_s: float = 3.0e-3,
) -> GridStudy:
    """Run the localisation pipeline over a square grid of source positions.

    Args:
        out_dir: directory for the injected ``.npz`` files (a temp dir in tests).
        grid_step: number of grid points along each axis (total = grid_step**2).
        grid_extent: (min, max) source coordinate in metres along each axis. The
            interior is used because the very edges (within the sensor inset) are
            where the ADR-006 failure mode is strongest.
        duration_s: simulation duration per event.

    Returns:
        A ``GridStudy`` with per-position outcomes and aggregate statistics.

    """
    sensors = np.array(SensorLayout().positions_m)
    grid = np.linspace(grid_extent[0], grid_extent[1], grid_step)
    positions: list[GridPosition] = []
    errors: list[float] = []

    for x in grid:
        for y in grid:
            path = inject_event("crack", float(x), float(y), duration_s=duration_s, out_dir=out_dir)
            with np.load(path) as data:
                wf = np.asarray(data["waveforms"])
                sr = float(data["sample_rate_hz"])
                src = np.asarray(data["source_xy_m"])

            arrival_times = pick_arrival_times(wf, sr, TdoaConfig())
            try:
                result = solve(sensors, arrival_times, C_EFF_M_PER_S)
            except ValueError:
                positions.append(
                    GridPosition(
                        source_xy_m=(float(x), float(y)),
                        estimated_xy_m=(float("nan"), float("nan")),
                        error_m=float("nan"),
                        residual_s=float("nan"),
                        confidence=float("nan"),
                    )
                )
                continue
            error = float(np.hypot(result.position_m[0] - src[0], result.position_m[1] - src[1]))
            errors.append(error)
            positions.append(
                GridPosition(
                    source_xy_m=(float(src[0]), float(src[1])),
                    estimated_xy_m=result.position_m,
                    error_m=error,
                    residual_s=result.residual,
                    confidence=result.confidence,
                )
            )

    if errors:
        e = np.array(errors)
        stats = dict(
            n_localised=len(errors),
            median_error_m=float(np.median(e)),
            mean_error_m=float(np.mean(e)),
            p90_error_m=float(np.percentile(e, 90)),
            max_error_m=float(np.max(e)),
        )
    else:
        stats = dict(
            n_localised=0,
            median_error_m=float("nan"),
            mean_error_m=float("nan"),
            p90_error_m=float("nan"),
            max_error_m=float("nan"),
        )

    return GridStudy(
        positions=positions,
        grid_x_m=grid,
        grid_y_m=grid,
        **stats,  # type: ignore[arg-type]
    )
