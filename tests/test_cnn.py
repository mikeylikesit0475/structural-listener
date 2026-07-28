"""Sprint 2.3: the CNN trains on the STFT front-end and produces finite accuracy.

We use the same small dataset as the baseline test (10/class train, 10/class test).
The point is to verify the pipeline runs end-to-end; the real accuracy number goes in
the README from a larger run (``model.train_cnn_cli``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from structural_listener.model.cnn import train_cnn
from structural_listener.model.dataset import generate_dataset


def test_cnn_trains_and_produces_finite_accuracy(tmp_path: Path) -> None:
    """The CNN trains and reports a finite accuracy >= 25% (chance on 4 classes)."""
    train = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=1)
    test = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=99)
    _model, report = train_cnn(train, test, epochs=10)
    acc = report["accuracy"]
    confusion = np.asarray(report["confusion"])
    assert np.isfinite(acc), f"accuracy not finite: {acc}"
    assert acc >= 0.25, f"CNN accuracy {acc:.2%} < 25% (chance) \u2014 not learning"
    assert confusion.shape == (4, 4), f"confusion shape {confusion.shape}"
