"""Sprint 2.6: confusion-matrix floor asserted in CI.

These tests are the regression gate: if a change to the feature extractor, dataset, or
classifier drops accuracy below the floor, the build fails. The floors are intentionally
below the README numbers (99.5% baseline, 76% CNN) so the gate catches *regressions*,
not noise from the small test dataset (10/class train, 10/class test).

The per-class recall check catches a class-collapse regression \u2014 e.g. if the crack
class collapses to noise (the CNN's known failure on the full dataset), the diagonal
entry for crack drops below 50% and the test fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from structural_listener.model.baseline import train_eval_baseline
from structural_listener.model.dataset import SOURCE_TYPES, generate_dataset


def test_baseline_confusion_matrix_floor(tmp_path: Path) -> None:
    """Baseline (logistic + forest) accuracy >= 60%, per-class recall >= 50%.

    The 60% / 50% floors are below the README numbers (99.5%) so the gate is robust to
    the small test dataset's stochasticity, but catches a real regression (e.g. a
    feature-extraction bug that breaks the duration separator).
    """
    train = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=1)
    test = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=99)
    for classifier in ("logistic", "forest"):
        report = train_eval_baseline(train, test, classifier=classifier)
        assert report.accuracy >= 0.60, f"{report.name} accuracy {report.accuracy:.2%} < 60% floor"
        # Per-class recall: each class must be correctly classified at least half the time.
        cm = np.asarray(report.confusion)
        for i, name in enumerate(SOURCE_TYPES):
            row_sum = int(cm[i].sum())
            if row_sum == 0:
                continue
            recall = cm[i, i] / row_sum
            assert recall >= 0.5, (
                f"{report.name}: {name} recall {recall:.0%} < 50% \u2014 class collapse"
            )
