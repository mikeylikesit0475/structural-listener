"""Sprint 0.6: the CLI writes a 4-channel .npz with ground-truth source coordinates.

The .npz must contain waveforms, sample rate, duration, source coordinates, source type,
sensor positions, gains, and distances \u2014 so downstream code never has to infer ground
truth (CLAUDE.md).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from structural_listener.cli.inject import inject_event


def _load_npz(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def test_inject_event_writes_valid_npz(tmp_path: Path) -> None:
    """The .npz has all required fields with correct shapes and ground truth."""
    out = inject_event("crack", 0.62, 0.31, duration_s=2.0e-3, out_dir=tmp_path)
    assert out.exists(), f"output file {out} not written"
    assert out.suffix == ".npz"

    d = _load_npz(out)
    # Required fields per ARCHITECTURE.md \u00a76 and CLAUDE.md.
    required = {
        "waveforms",
        "sample_rate_hz",
        "duration_s",
        "source_xy_m",
        "source_type",
        "sensor_positions_m",
        "gains",
        "distances_m",
    }
    assert required.issubset(d.keys()), f"missing fields: {required - set(d.keys())}"

    # Waveforms: 4 channels, n_samples > 0.
    wf = np.asarray(d["waveforms"])
    assert wf.ndim == 2 and wf.shape[0] == 4 and wf.shape[1] > 0

    # Ground truth: the coordinates we asked for.
    src = np.asarray(d["source_xy_m"])
    np.testing.assert_allclose(src, [0.62, 0.31], atol=1e-9)

    # Source type is a string label.
    assert str(d["source_type"]) == "crack"

    # Sensor positions: 4 corners.
    sp = np.asarray(d["sensor_positions_m"])
    assert sp.shape == (4, 2)

    # Sample rate and duration are positive scalars.
    assert float(d["sample_rate_hz"]) > 0.0
    assert float(d["duration_s"]) > 0.0


def test_inject_rejects_out_of_plate_source(tmp_path: Path) -> None:
    """A source outside the plate raises (don't silently write bad ground truth)."""
    with pytest.raises(ValueError, match="outside plate"):
        inject_event("crack", 1.5, 0.5, out_dir=tmp_path)


def test_inject_rejects_unknown_source_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown source type"):
        inject_event(" earthquake", 0.5, 0.5, out_dir=tmp_path)


def test_make_target_crack_writes_npz(tmp_path: Path) -> None:
    """``make crack X= Y=`` runs end-to-end via the Makefile."""
    # We invoke the CLI module directly rather than shelling out to make, so the test
    # doesn't depend on having GNU make installed. The Makefile target is a thin wrapper
    # around this exact command; we test the command, not the wrapper.
    cmd = [
        sys.executable,
        "-m",
        "structural_listener.cli.inject",
        "crack",
        "--x",
        "0.62",
        "--y",
        "0.31",
        "--duration",
        "1.5e-3",
        "--out-dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    npz_files = list(tmp_path.glob("crack_*.npz"))
    assert len(npz_files) == 1, f"expected 1 .npz, found {len(npz_files)}"


def test_all_four_source_types_inject(tmp_path: Path) -> None:
    """All four signatures from the registry produce a valid .npz."""
    for src_type in ("crack", "friction", "impact", "noise"):
        out = inject_event(src_type, 0.5, 0.5, duration_s=1.0e-3, out_dir=tmp_path)
        assert out.exists(), f"{src_type} did not write a file"
        d = _load_npz(out)
        assert str(d["source_type"]) == src_type
        wf = np.asarray(d["waveforms"])
        assert wf.shape[0] == 4
