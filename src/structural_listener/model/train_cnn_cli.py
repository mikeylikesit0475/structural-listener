"""Train the CNN classifier and print the confusion matrix (Sprint 2.3).

Run: ``uv run python -m structural_listener.model.train_cnn_cli``

Trains the small CNN on the STFT front-end, prints accuracy + confusion matrix. This
number goes in the README side-by-side with the baseline (ADR-004). The honest
comparison is in ``docs/interview-stories/baseline-vs-cnn.md``.

This is a slow script (~5 min for dataset generation + training). Not run in CI; the
CI floor assertion (2.6) uses the smaller dataset in ``tests/test_cnn.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from structural_listener.model.cnn import train_cnn
from structural_listener.model.dataset import SOURCE_TYPES, generate_dataset


def main() -> int:
    """Train the CNN and print the result."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("generating training dataset (100/class)...")
        train = generate_dataset(n_per_class=100, out_dir=tmp_path, seed=1)
        print("generating test dataset (50/class)...")
        test = generate_dataset(n_per_class=50, out_dir=tmp_path, seed=99)

    _model, report = train_cnn(train, test, epochs=30)
    acc = float(report["accuracy"])
    confusion = np.asarray(report["confusion"])
    print(f"\nCNN (STFT): accuracy = {acc:.1%}")
    print("  confusion matrix (rows=true, cols=pred):")
    print(f"    {'':>10}" + "".join(f"{s:>10}" for s in SOURCE_TYPES))
    for i, name in enumerate(SOURCE_TYPES):
        row = confusion[i]
        print(f"    {name:>10}" + "".join(f"{int(c):>10}" for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
