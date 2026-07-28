"""CLI: inject a simulated event and write a 4-channel ``.npz`` with ground truth.

Invoked by the Makefile targets ``crack``, ``friction``, ``impact``, ``noise``:

    make crack X=0.62 Y=0.31
    make friction X=0.5 Y=0.5
    make impact X=0.3 Y=0.7
    make noise X=0.5 Y=0.5

Each target runs ``uv run python -m structural_listener.cli.inject <type> --x X --y Y``
and writes ``data/out/<type>_<x>_<y>.npz`` containing:

- ``waveforms``: (4, n_samples) sensor signals in m/s, after the sensor model.
- ``sample_rate_hz``: sampling rate.
- ``duration_s``: simulated physics time.
- ``source_xy_m``: ground-truth source coordinates (x, y) in metres.
- ``source_type``: the event label string.
- ``sensor_positions_m``: (4, 2) array of sensor coordinates.
- ``gains``: per-channel gains applied by the sensor model.
- ``distances_m``: per-channel source-to-sensor distances.

Ground truth travels with the data (CLAUDE.md) \u2014 never inferred downstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from structural_listener.sim.config import PlateConfig, SensorLayout, SolverConfig
from structural_listener.sim.fdtd import Source, simulate
from structural_listener.sim.sensor_model import SensorModelConfig, apply_sensor_model
from structural_listener.sim.signatures import SOURCE_REGISTRY


def _build_source(source_type: str, x: float, y: float, dx: float, nx: int, ny: int) -> Source:
    cls = SOURCE_REGISTRY.get(source_type)
    if cls is None:
        raise ValueError(
            f"unknown source type {source_type!r}; choose one of {sorted(SOURCE_REGISTRY)}"
        )
    source: Source = cls((x, y), dx, nx, ny)
    return source


def inject_event(
    source_type: str,
    x: float,
    y: float,
    duration_s: float = 5.0e-3,
    out_dir: Path | None = None,
) -> Path:
    """Run a single event simulation and write the result to an ``.npz`` file.

    Returns the path to the written file.
    """
    plate = PlateConfig()
    solver = SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx

    if not (0.0 <= x <= plate.length and 0.0 <= y <= plate.width):
        raise ValueError(
            f"source ({x}, {y}) outside plate [0, {plate.length}] x [0, {plate.width}]"
        )

    source = _build_source(source_type, x, y, dx, nx, ny)
    sim_result = simulate(source, duration_s=duration_s, plate=plate, solver=solver)
    waveforms, meta = apply_sensor_model(sim_result, config=SensorModelConfig())

    out_dir = out_dir or Path("data/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_type}_{x:.3f}_{y:.3f}.npz"

    np.savez(
        out_path,
        waveforms=waveforms,
        sample_rate_hz=sim_result.sample_rate_hz,
        duration_s=sim_result.duration_s,
        source_xy_m=np.array(sim_result.true_source_xy_m, dtype=np.float64),
        source_type=source_type,
        sensor_positions_m=np.array(SensorLayout().positions_m, dtype=np.float64),
        gains=meta["gains"],
        distances_m=meta["distances_m"],
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m structural_listener.cli.inject``."""
    parser = argparse.ArgumentParser(
        description="Inject a simulated acoustic-emission event and write a 4-channel .npz."
    )
    parser.add_argument(
        "source_type",
        choices=sorted(SOURCE_REGISTRY),
        help="event type to inject",
    )
    parser.add_argument("--x", type=float, required=True, help="source x coordinate (m)")
    parser.add_argument("--y", type=float, required=True, help="source y coordinate (m)")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0e-3,
        help="simulation duration in seconds (default 5 ms)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/out"),
        help="output directory (default data/out)",
    )
    args = parser.parse_args(argv)

    path = inject_event(
        args.source_type, args.x, args.y, duration_s=args.duration, out_dir=args.out_dir
    )
    print(
        f"injected {args.source_type} at ({args.x:.3f}, {args.y:.3f}) m -> {path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
