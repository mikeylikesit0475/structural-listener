"""Regression: a source must be injected at the requested grid cell, not the corner.

This guards against the misplaced-paren bug
``min(max(v, 1, nx-2), 1)`` that pinned every event at grid cell (1, 1) regardless of
the requested ``(x, y)``. The symptoms were identical picks on channels 1 and 3 (the
field was symmetric about the grid diagonal) and a median localisation error equal to
the average distance from a grid point to the bottom-left corner.

We assert two things:
1. The injected grid cell ``(i, j)`` matches ``round(source_xy_m / dx)`` for several
   source positions across the plate (clamped to the interior).
2. The four channel waveforms for two *different* source positions differ \u2014 the
   buggy code produced identical waveforms because the physical source never moved.
"""

from __future__ import annotations

import numpy as np

from structural_listener.sim.config import PlateConfig, SolverConfig
from structural_listener.sim.fdtd import simulate
from structural_listener.sim.signatures import CrackGrowthSource


def _injected_cell(source_xy_m: tuple[float, float]) -> tuple[int, int]:
    """Return the (i, j) grid cell the source actually excites."""
    # The _BaseSource clamps to the interior grid; mirror that here.
    solver = SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx
    i = min(max(round(source_xy_m[0] / dx), 1), nx - 2)
    j = min(max(round(source_xy_m[1] / dx), 1), ny - 2)
    return (i, j)


def test_source_grid_cell_matches_requested_coordinates() -> None:
    """The injected source cell equals round(xy/dx) for several positions."""
    solver = SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx

    for x in (0.10, 0.30, 0.50, 0.70, 0.90):
        for y in (0.15, 0.45, 0.85):
            src = CrackGrowthSource((x, y), dx, nx, ny)
            expected = (
                min(max(round(x / dx), 1), nx - 2),
                min(max(round(y / dx), 1), ny - 2),
            )
            assert (src._i, src._j) == expected, (
                f"source ({x},{y}) injected at ({src._i},{src._j}) "
                f"expected {expected} \u2014 injection-cell bug regressed"
            )


def test_different_source_positions_produce_different_waveforms() -> None:
    """Two distinct source positions must produce distinct sensor waveforms.

    The injection-cell bug made every event originate at (1,1), so the waveforms
    depended only on the requested duration, not the position. This test fails loudly
    if that regresses: we compare channel 0 of an off-centre event to a centre event
    and require a non-trivial difference.
    """
    solver = SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx
    plate = PlateConfig()

    sim_a = simulate(CrackGrowthSource((0.20, 0.20), dx, nx, ny), duration_s=1.5e-3, plate=plate)
    sim_b = simulate(CrackGrowthSource((0.80, 0.80), dx, nx, ny), duration_s=1.5e-3, plate=plate)
    # Channel 0 is near (0.05, 0.05); the (0.20,0.20) source is much closer to it
    # than the (0.80,0.80) source, so the early-time energy must differ.
    early = slice(0, int(0.5e-3 * sim_a.sample_rate_hz))
    e_a = float(np.sum(sim_a.waveforms[0, early] ** 2))
    e_b = float(np.sum(sim_b.waveforms[0, early] ** 2))
    assert e_a > 1e3 * e_b, (
        f"near-source event energy {e_a:.2e} not >> far-source {e_b:.2e} on ch0 "
        f"\u2014 source position is not affecting the waveform (injection bug?)"
    )
