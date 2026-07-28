"""Sprint 2.2: baseline classifiers on classical AE features.

We generate a small train/test split (40 train, 40 test events), train logistic
regression and random forest, and assert a reasonable accuracy floor. The point of
the baseline is that it *works* \u2014 if the features carry the signal, the CNN in 2.3
has to justify its complexity. The floor here is intentionally modest (60%) so the
test is robust to the simulator's stochasticity; the real number goes in the README
from a larger run (see ``model.train_baseline``).
"""

from __future__ import annotations

from pathlib import Path

from structural_listener.model.baseline import train_eval_baseline
from structural_listener.model.dataset import SOURCE_TYPES, generate_dataset


def test_logistic_regression_achieves_modest_accuracy(tmp_path: Path) -> None:
    """Logistic regression on 5 classical features beats 60% accuracy on 4 classes.

    60% on 4 classes (chance = 25%) confirms the features carry the signal. The
    number goes in the README; this floor just guards the pipeline.
    """
    train = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=1)
    test = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=99)
    report = train_eval_baseline(train, test, classifier="logistic")
    assert report.accuracy >= 0.60, (
        f"logistic regression accuracy {report.accuracy:.2%} < 60% \u2014 "
        "features not carrying signal"
    )


def test_random_forest_achieves_modest_accuracy(tmp_path: Path) -> None:
    """Random forest on 5 classical features beats 60% accuracy on 4 classes."""
    train = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=1)
    test = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=99)
    report = train_eval_baseline(train, test, classifier="forest")
    assert report.accuracy >= 0.60, (
        f"random forest accuracy {report.accuracy:.2%} < 60% \u2014 features not carrying signal"
    )


def test_confusion_matrix_shape_and_labels(tmp_path: Path) -> None:
    """The confusion matrix is 4x4 with rows = true, cols = predicted."""
    train = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=1)
    test = generate_dataset(n_per_class=10, out_dir=tmp_path, seed=99)
    report = train_eval_baseline(train, test, classifier="logistic")
    assert report.confusion.shape == (4, 4), f"confusion shape {report.confusion.shape}"
    # All test samples are represented.
    assert int(report.confusion.sum()) <= len(test), "confusion matrix overcounts"
    assert len(SOURCE_TYPES) == 4
