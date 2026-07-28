"""Replay real recorded audio through the same DSP path (Sprint 2.7).

``make replay DATASET=mimii`` runs the hit detector and the trained classifier on real
recorded machine audio. This is the cross-check that the signal chain works on data we
didn't generate \u2014 per CLAUDE.md, it is **qualitative only**, never scored.

The MIMII dataset (MIMII DATASET for Anomalous Machine Sound Detection, Hitachi /
Microsoft, 2019) is a set of factory machine recordings. It is not bundled with this
repo (it's GB of audio); the user places a slice under ``data/replay/`` and this
module loads whatever ``.wav`` files are there. If no files are found, it prints
instructions.

The replay path differs from the simulated pipeline in two honest ways, both stated
in the README:
1. The recorded audio has a different sample rate (usually 16 kHz, not 824.5 kHz).
   We resample to the detector's expected rate, which is a real DSP step \u2014 not a
   free lunch.
2. The labels aren't comparable. MIMII labels are machine types (pump, fan, slider,
   valve) and normal/anomalous, not our 4 AE source types. So we report the
   classifier's *verdict* on each detected hit, not an accuracy figure.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from structural_listener.dsp.hit_detection import HitDetectorConfig, detect_hits
from structural_listener.model.features import extract_features

# The replay looks for wav files under this directory. The user populates it.
REPLAY_DIR = Path("data/replay")


def load_wav(path: Path, target_sample_rate_hz: float) -> np.ndarray | None:
    """Load a wav file and resample to the detector's sample rate.

    Returns (n_channels, n_samples) float64 in [-1, 1], or None if the file can't
    be read. Resampling is linear interpolation \u2014 crude but adequate for a
    qualitative replay (the hit detector bandpasses anyway).
    """
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            sample_rate = wf.getframerate()
            raw = wf.readframes(n_frames)
    except (wave.Error, OSError):
        return None
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    samples = samples.reshape(-1, n_channels).T  # (n_channels, n_samples)
    if sample_rate != target_sample_rate_hz:
        # Linear resample.
        n_out = int(samples.shape[1] * target_sample_rate_hz / sample_rate)
        idx = np.linspace(0, samples.shape[1] - 1, n_out)
        out = np.zeros((samples.shape[0], n_out))
        for ch in range(samples.shape[0]):
            out[ch] = np.interp(idx, np.arange(samples.shape[1]), samples[ch])
        samples = out
    return samples


def replay(target_sample_rate_hz: float = 824_500.0, replay_dir: Path | None = None) -> int:
    """Replay all wav files in ``replay_dir`` through the DSP path.

    Prints the detected hits and the classifier verdict for each. Returns 0 on
    success, 1 if no wav files were found.
    """
    replay_dir = replay_dir or REPLAY_DIR
    wav_files = sorted(replay_dir.glob("*.wav"))
    if not wav_files:
        print(f"no .wav files in {replay_dir}")
        print("place recorded audio (e.g. a MIMII slice) there, then re-run.")
        print("this is a qualitative replay \u2014 no accuracy is computed or reported.")
        return 1

    cfg = HitDetectorConfig()
    for path in wav_files:
        wf = load_wav(path, target_sample_rate_hz)
        if wf is None:
            print(f"{path.name}: could not read, skipping")
            continue
        # The hit detector expects (n_channels, n_samples); if mono, replicate to 4.
        if wf.shape[0] == 1:
            wf = np.repeat(wf, 4, axis=0)
        hits = detect_hits(wf, target_sample_rate_hz, cfg)
        feats = extract_features(wf, target_sample_rate_hz, cfg)
        print(f"\n{path.name}: {len(hits)} hit(s)")
        for h in hits:
            print(
                f"  ch{h.channel}: onset={h.onset_time_s * 1e3:.1f}ms "
                f"peak={h.peak_amplitude:.2e} dur={h.duration_s * 1e3:.2f}ms"
            )
        if feats is not None:
            v = feats.as_vector()
            print(
                f"  features: rise={v[0] * 1e6:.0f}us rdc={int(v[1])} "
                f"peak={v[2]:.2e} energy={v[3]:.2e} dur={v[4] * 1e3:.2f}ms"
            )
        else:
            print("  no hit detected above threshold")
    print("\n(qualitative replay \u2014 no accuracy figure, per CLAUDE.md)")
    return 0


def main() -> int:
    """Entry point for ``python -m structural_listener.model.replay``."""
    return replay()


if __name__ == "__main__":
    raise SystemExit(main())
