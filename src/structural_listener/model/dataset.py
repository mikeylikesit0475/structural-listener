"""Training dataset generation for the classifier (Sprint 2.2).

Generates a labelled dataset of simulated AE events: N events per source type, each at
a random plate position, run through the sensor model. The classical feature extractor
(``model.features``) produces one feature vector per event; the labels are the four
source types.

This is the dataset the Sprint 2.2 baseline (logistic regression + random forest) and
the Sprint 2.3 CNN both train on. The dataset is *simulated* \u2014 the README says so
in the first paragraph, and the accuracy validates the pipeline, not the physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from structural_listener.cli.inject import inject_event
from structural_listener.model.features import AeFeatures, extract_features

SOURCE_TYPES: tuple[str, ...] = ("crack", "friction", "impact", "noise")
LABEL_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(SOURCE_TYPES)}


@dataclass(frozen=True, slots=True)
class EventSample:
    """One labelled event in the dataset.

    Attributes:
        source_type: the event label (one of SOURCE_TYPES).
        label: integer label (index into SOURCE_TYPES).
        features: classical AE feature vector (5,), or None if no hit was detected.
        waveforms: (4, n_samples) sensor signals, for the STFT/CNN front-end.
        sample_rate_hz: sampling rate.
        source_xy_m: ground-truth source position (metres).

    """

    source_type: str
    label: int
    features: AeFeatures | None
    waveforms: np.ndarray
    sample_rate_hz: float
    source_xy_m: tuple[float, float]


def generate_dataset(
    n_per_class: int,
    out_dir: Path,
    seed: int = 42,
    plate_extent: tuple[float, float] = (0.15, 0.85),
    noise_duration_s: float = 10.0e-3,
) -> list[EventSample]:
    """Generate a labelled dataset of simulated AE events.

    Args:
        n_per_class: number of events per source type.
        out_dir: directory for the injected ``.npz`` files.
        seed: RNG seed for reproducible source positions.
        plate_extent: (min, max) source coordinate; the interior avoids the
            reflection-dominated edge cases (ADR-006) so the classifier sees
            representative direct-arrival signals.
        noise_duration_s: simulation duration for the electrical-noise class. The
            50 Hz mains signal needs several cycles to register a detectable hit;
            10 ms (5 cycles) is the minimum that works. Longer is more realistic
            but the FDTD solver scales linearly with duration.

    Returns:
        A list of ``EventSample``, one per event, in SOURCE_TYPES order.

    """
    rng = np.random.default_rng(seed)
    samples: list[EventSample] = []
    for source_type in SOURCE_TYPES:
        duration_s = noise_duration_s if source_type == "noise" else 4.0e-3
        for _ in range(n_per_class):
            x = float(rng.uniform(*plate_extent))
            y = float(rng.uniform(*plate_extent))
            path = inject_event(source_type, x, y, duration_s=duration_s, out_dir=out_dir)
            with np.load(path) as data:
                wf = np.asarray(data["waveforms"])
                sr = float(data["sample_rate_hz"])
                src_xy = (float(data["source_xy_m"][0]), float(data["source_xy_m"][1]))
            feats = extract_features(wf, sr)
            samples.append(
                EventSample(
                    source_type=source_type,
                    label=LABEL_TO_INDEX[source_type],
                    features=feats,
                    waveforms=wf,
                    sample_rate_hz=sr,
                    source_xy_m=src_xy,
                )
            )
    return samples


def feature_matrix(samples: list[EventSample]) -> tuple[np.ndarray, np.ndarray]:
    """Extract the (X, y) classification matrix from samples.

    Samples with no detected hit (features is None) are dropped \u2014 the classifier
    can't classify what it didn't detect. The matrix X has the 5 classical features in
    ``FEATURE_NAMES`` order, log-scaled to tame the wide dynamic range (peak amplitude
    and energy span many orders of magnitude). y is the integer label.

    Returns (X, y) as float64 / int64 arrays.
    """
    from structural_listener.model.features import FEATURE_NAMES

    _ = FEATURE_NAMES  # the order is documented there
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for s in samples:
        if s.features is None:
            continue
        v = s.features.as_vector()
        # Log-scale the positive features (all 5 are >= 0). Adding 1e-30 avoids log(0).
        # This puts peak amplitude (1e-9 to 1e-5) and energy (1e-22 to 1e-13) on the same
        # scale as rise time (1e-5 to 1e-3) and duration (1e-4 to 1e-3) for the linear model.
        log_v = np.log10(v + 1.0e-30)
        rows.append(log_v)
        labels.append(s.label)
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)
