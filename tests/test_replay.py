"""Sprint 2.7: replay module loads wav files and runs the DSP path.

Qualitative-only tests: we verify the replay handles an empty directory gracefully
and processes a synthetic wav file. We do NOT assert any accuracy figure \u2014 per
CLAUDE.md, real recorded data is qualitative only.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from structural_listener.model.replay import load_wav, replay


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a mono int16 wav."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())


def test_replay_empty_dir_returns_one(tmp_path: Path) -> None:
    """An empty replay directory prints instructions and returns 1."""
    rc = replay(replay_dir=tmp_path)
    assert rc == 1, "empty replay dir should return 1"


def test_load_wav_reads_and_resamples(tmp_path: Path) -> None:
    """load_wav reads a wav and resamples to the target rate."""
    sr_in = 16000
    sr_out = 824_500.0
    t = np.arange(int(0.1 * sr_in)) / sr_in
    sig = 0.5 * np.sin(2 * np.pi * 1000 * t)
    wav_path = tmp_path / "test.wav"
    _write_wav(wav_path, sig, sr_in)
    wf = load_wav(wav_path, sr_out)
    assert wf is not None
    assert wf.shape[0] == 1, f"expected mono, got {wf.shape[0]} channels"
    # Resampled length should be ~0.1s * sr_out.
    assert abs(wf.shape[1] - int(0.1 * sr_out)) < 100, f"resampled length {wf.shape[1]} wrong"


def test_replay_processes_synthetic_wav(tmp_path: Path) -> None:
    """replay runs the DSP path on a synthetic wav and returns 0."""
    # A 50 ms 16 kHz wav with a short burst \u2014 should produce output without crashing.
    sr_in = 16000
    t = np.arange(int(0.05 * sr_in)) / sr_in
    sig = 0.1 * np.sin(2 * np.pi * 2000 * t)
    wav_path = tmp_path / "synthetic.wav"
    _write_wav(wav_path, sig, sr_in)
    rc = replay(replay_dir=tmp_path)
    assert rc == 0, "replay with valid wav should return 0"
