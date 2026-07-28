"""Generate ``docs/figures/demo.gif`` \u2014 an animated simulation of the live UI.

Renders a matplotlib animation that mimics what the Next.js UI shows during the demo:
1. The plate with four sensors.
2. The true source position (green cross).
3. The four waveforms arriving in real time (animated left-to-right).
4. The localisation estimate appearing (blue circle) with the error readout in mm.

This is a *simulation* of the UI, not a screen recording \u2014 the real demo is the live
UI. The GIF is for the README and the demo script's pre-flight reference.

Run: ``uv run python -m structural_listener.eval.plot_demo_gif``
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from typing import Any, cast

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from structural_listener.cli.inject import inject_event
from structural_listener.dsp.tdoa import TdoaConfig, pick_arrival_times
from structural_listener.localize.multilateration import solve

_OUT_PATH = Path("docs/figures/demo.gif")

# Effective wave speed (docs/sources.md).
_C_EFF = 1200.0

# Animation settings.
_N_FRAMES = 120
_FPS = 30


def _make_figure() -> tuple[Figure, Axes, Axes]:
    """Create the figure: plate on the left, waveforms on the right."""
    fig = plt.figure(figsize=(10, 4.5), dpi=100)
    ax_plate = fig.add_axes((0.05, 0.1, 0.42, 0.8))
    ax_wave = fig.add_axes((0.55, 0.1, 0.4, 0.8))
    return fig, ax_plate, ax_wave


def _draw_plate(
    ax: Axes,
    sensors: np.ndarray,
    src: np.ndarray,
    est: np.ndarray | None,
    error_mm: float | None,
    show_est: bool,
) -> list[object]:
    """Draw the plate, sensors, true source, and (optionally) the estimate."""
    artists: list[object] = []
    # Plate outline.
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f8f8")
    artists.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="#444", lw=2))
    ax.add_patch(artists[-1])  # type: ignore[arg-type]
    # Sensors.
    for i, (sx, sy) in enumerate(sensors):
        c = Circle((sx, sy), 0.025, color="red", zorder=5)
        ax.add_patch(c)
        artists.append(c)
        t = ax.text(sx + 0.03, sy + 0.03, f"s{i}", fontsize=7, color="#444")
        artists.append(t)
    # True source (green cross).
    tx, ty = src
    artists.append(ax.plot([tx - 0.03, tx + 0.03], [ty, ty], "g-", lw=2)[0])
    artists.append(ax.plot([tx, tx], [ty - 0.03, ty + 0.03], "g-", lw=2)[0])
    # Estimated position (blue circle) — only when show_est is True.
    if show_est and est is not None:
        ex, ey = est
        artists.append(ax.plot([ex], [ey], "bo", ms=10, fillstyle="none", lw=2)[0])
        # Error line.
        artists.append(ax.plot([ex, tx], [ey, ty], "k--", lw=1, alpha=0.5)[0])
        # Error label.
        if error_mm is not None:
            artists.append(
                ax.text(
                    0.5,
                    -0.08,
                    f"error: {error_mm} mm",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                    color="blue",
                    transform=ax.transAxes,
                )
            )
    ax.set_title("Plate (green=true, blue=estimate)", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return artists


def _draw_waveforms(
    ax: Axes,
    waveforms: np.ndarray,
    sr: float,
    progress: float,
) -> list[object]:
    """Draw the four waveforms up to ``progress`` fraction of the total duration."""
    artists: list[object] = []
    n_total = waveforms.shape[1]
    n_show = int(n_total * progress)
    ch_h = 0.22
    for ch in range(4):
        offset = ch * ch_h + 0.5
        seg = waveforms[ch, :n_show]
        if seg.shape[0] < 2:
            continue
        t = np.arange(n_show) / sr * 1e3  # ms
        max_abs = max(float(np.max(np.abs(waveforms[ch]))), 1e-12)
        y = seg / max_abs * (ch_h * 0.35)
        color = plt.cm.tab10(ch / 10.0)
        artists.append(ax.plot(t, y + offset, color=color, lw=0.5)[0])
        ax.text(-0.5, offset + ch_h * 0.3, f"ch{ch}", fontsize=7, color=color, va="top")
    ax.set_xlim(0, n_total / sr * 1e3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Waveforms", fontsize=10)
    ax.set_xlabel("time (ms)")
    ax.set_yticks([])
    return artists


def main() -> int:
    """Generate the demo GIF."""
    with tempfile.TemporaryDirectory() as tmp:
        # Inject a crack event at a visible position.
        path = inject_event("crack", 0.62, 0.31, duration_s=3.0e-3, out_dir=Path(tmp))
        with np.load(path) as data:
            wf = np.asarray(data["waveforms"])
            sr = float(data["sample_rate_hz"])
            src = np.asarray(data["source_xy_m"])
            sensors = np.asarray(data["sensor_positions_m"])

        # Compute the localisation.
        arrival_times = pick_arrival_times(wf, sr, TdoaConfig())
        result = solve(sensors, arrival_times, _C_EFF)
        est = np.array(result.position_m)
        error_mm = round(float(np.hypot(est[0] - src[0], est[1] - src[1]) * 1000))

    fig, ax_plate, ax_wave = _make_figure()

    def update(frame: int) -> list[object]:
        """One animation frame: plate + waveforms at ``progress`` fraction."""
        progress = (frame + 1) / _N_FRAMES
        # Clear and redraw (simple approach for a short animation).
        ax_plate.clear()
        ax_wave.clear()
        show_est = frame >= _N_FRAMES * 0.7  # estimate appears at 70% through
        a1 = _draw_plate(ax_plate, sensors, src, est, error_mm, show_est)
        a2 = _draw_waveforms(ax_wave, wf, sr, progress)
        return a1 + a2

    anim = animation.FuncAnimation(
        fig,
        cast(Any, update),
        frames=_N_FRAMES,
        interval=1000 / _FPS,
        blit=False,
    )
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(_OUT_PATH, writer="pillow", fps=_FPS)
    plt.close(fig)
    print(f"wrote {_OUT_PATH} ({_N_FRAMES} frames at {_FPS} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
