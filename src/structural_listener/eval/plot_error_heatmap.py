"""Generate ``docs/figures/error-heatmap.png`` \u2014 the Sprint 1.4 error distribution.

Run: ``uv run python -m structural_listener.eval.plot_error_heatmap``

Writes the heat map and a summary line to stdout. The heat map shows localisation
error in cm vs. source position on the 1 m plate, with sensor positions marked. The
ADR-006 failure mode is visible as higher error near the plate edges, where the
direct-to-first-reflection gap is smallest.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from structural_listener.eval.localisation_grid import GridStudy, run_grid_study
from structural_listener.sim.config import SensorLayout

_OUT_PATH = Path("docs/figures/error-heatmap.png")


def _plot(study: GridStudy, out_path: Path) -> None:
    n = study.grid_x_m.size
    # The grid is row-major over (x, y); reshape error into (n_y, n_x) for imshow.
    # imshow expects [row, col] = [y, x]; we want y increasing upward, so flip.
    error_grid = np.full((n, n), np.nan)
    for k, pos in enumerate(study.positions):
        i = k % n  # x index
        j = k // n  # y index
        error_grid[j, i] = pos.error_m
    error_grid_cm = error_grid * 100.0  # m -> cm

    sensors = np.array(SensorLayout().positions_m)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    # Heat map: origin lower-left so y increases upward.
    im = ax.imshow(
        error_grid_cm,
        origin="lower",
        extent=(
            study.grid_x_m[0],
            study.grid_x_m[-1],
            study.grid_y_m[0],
            study.grid_y_m[-1],
        ),
        cmap="viridis_r",  # reversed: dark = low error (good)
        vmin=0.0,
        vmax=float(np.nanpercentile(error_grid_cm, 95)),  # clip outliers for contrast
        aspect="equal",
    )
    ax.scatter(
        sensors[:, 0], sensors[:, 1], marker="o", s=80, c="red", edgecolors="white", zorder=5
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Localisation error (cm) \u2014 median {study.median_error_m * 100:.1f} cm, "
        f"p90 {study.p90_error_m * 100:.1f} cm, "
        f"max {study.max_error_m * 100:.1f} cm"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("error (cm)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Entry point: run the study and write the heat map PNG."""
    with tempfile.TemporaryDirectory() as tmp:
        study = run_grid_study(Path(tmp))
    _plot(study, _OUT_PATH)
    print(
        f"n={study.n_localised}/{len(study.positions)}, "
        f"median={study.median_error_m * 100:.1f}cm, "
        f"mean={study.mean_error_m * 100:.1f}cm, "
        f"p90={study.p90_error_m * 100:.1f}cm, "
        f"max={study.max_error_m * 100:.1f}cm"
    )
    print(f"wrote {_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
