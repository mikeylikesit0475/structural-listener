"""Classical baseline classifiers for AE source classification (Sprint 2.2).

Two baselines, per ARCHITECTURE.md \u00a75:
- **Logistic regression** on the 5 classical AE features. The simplest linear model \u2014
  if this works, the features carry the signal and the CNN is overkill.
- **Random forest** on the same features. A non-linear baseline that's the standard
  tabular-classification workhorse.

Both numbers go in the README regardless of what the CNN achieves (ADR-004). If the
baseline is within a point of the CNN, say so \u2014 that's the more interesting result.

The classifiers are trained on the simulated dataset (``model.dataset``) with a fixed
random seed for reproducibility. The confusion matrix and accuracy are returned for
the honest comparison writeup (2.4) and the CI floor assertion (2.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from structural_listener.model.dataset import SOURCE_TYPES, EventSample, feature_matrix


@dataclass(frozen=True, slots=True)
class ClassifierReport:
    """The result of training + evaluating one classifier.

    Attributes:
        name: human-readable classifier name.
        accuracy: fraction correct on the test set.
        confusion: (n_classes, n_classes) confusion matrix; rows are true, cols pred.
        predictions: predicted labels for the test set.

    """

    name: str
    accuracy: float
    confusion: np.ndarray
    predictions: np.ndarray


def train_eval_baseline(
    train_samples: list[EventSample],
    test_samples: list[EventSample],
    classifier: str = "logistic",
    seed: int = 42,
) -> ClassifierReport:
    """Train a baseline classifier on classical AE features and evaluate on the test set.

    Args:
        train_samples: training events (from ``generate_dataset``).
        test_samples: held-out test events.
        classifier: "logistic" or "forest".
        seed: RNG seed for reproducibility.

    Returns:
        A ``ClassifierReport`` with accuracy and confusion matrix.

    """
    x_train, y_train = feature_matrix(train_samples)
    x_test, y_test = feature_matrix(test_samples)
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        raise ValueError("empty feature matrix \u2014 no hits detected in the dataset")

    if classifier == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        name = "logistic regression"
    elif classifier == "forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        name = "random forest"
    else:
        raise ValueError(f"unknown classifier {classifier!r}; use 'logistic' or 'forest'")

    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds, labels=list(range(len(SOURCE_TYPES))))
    return ClassifierReport(name=name, accuracy=acc, confusion=cm, predictions=preds)
