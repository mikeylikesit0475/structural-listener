"""Train and evaluate the classical baseline classifiers; record the confusion matrix.

Run: ``uv run python -m structural_listener.model.train_baseline``

Generates a train/test split (100 per class for training, 50 for testing), trains
logistic regression and random forest on the 5 classical AE features, and prints
accuracy + confusion matrix for each. These are the numbers that go in the README
regardless of what the CNN achieves (ADR-004).

This is a slow script (~5 min for dataset generation). It is not run in CI; the CI
floor assertion (2.6) uses the smaller dataset in ``tests/test_baseline.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from structural_listener.model.baseline import train_eval_baseline
from structural_listener.model.dataset import SOURCE_TYPES, generate_dataset


def main() -> int:
    """Train both baselines and print the results."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("generating training dataset (100/class)...")
        train = generate_dataset(n_per_class=100, out_dir=tmp_path, seed=1)
        print("generating test dataset (50/class)...")
        test = generate_dataset(n_per_class=50, out_dir=tmp_path, seed=99)

    for classifier in ("logistic", "forest"):
        report = train_eval_baseline(train, test, classifier=classifier)
        print(f"\n{report.name}: accuracy = {report.accuracy:.1%}")
        print("  confusion matrix (rows=true, cols=pred):")
        print(f"    {'':>10}" + "".join(f"{s:>10}" for s in SOURCE_TYPES))
        for i, name in enumerate(SOURCE_TYPES):
            row = report.confusion[i]
            print(f"    {name:>10}" + "".join(f"{int(c):>10}" for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
