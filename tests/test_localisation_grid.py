"""Sprint 1.4: the error-distribution study runs and reports finite statistics.

We use a small 4x4 grid here (the full 10x10 is exercised by
``test_multilateration.test_median_error_under_5cm_over_100_positions``). The point is
to guard the ``run_grid_study`` interface and confirm the interior median is within
the Sprint 1 target.
"""

from __future__ import annotations

import math
from pathlib import Path

from structural_listener.eval.localisation_grid import run_grid_study


def test_grid_study_runs_and_reports_finite_stats(tmp_path: Path) -> None:
    study = run_grid_study(tmp_path, grid_step=4, grid_extent=(0.25, 0.75))
    assert study.n_localised == 16, f"expected 16 localisations, got {study.n_localised}"
    assert math.isfinite(study.median_error_m), "median error not finite"
    assert math.isfinite(study.p90_error_m), "p90 error not finite"
    # Interior positions should comfortably beat the 5 cm target.
    assert study.median_error_m < 0.05, (
        f"interior median {study.median_error_m * 100:.1f} cm > 5 cm"
    )
    # Every position has an error recorded (none should be NaN in the interior).
    assert all(math.isfinite(p.error_m) for p in study.positions), "interior NaN error"
