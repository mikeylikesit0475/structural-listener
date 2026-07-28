"""Small CNN classifier on the STFT front-end (Sprint 2.3).

A deliberately small CNN: two conv blocks + a linear head. The point is not to win a
Kaggle competition \u2014 it's to compare honestly against the 5-feature logistic
regression baseline (ADR-004). If the baseline ties or wins, that's the stronger
interview answer.

Architecture: (F_BINS x N_FRAMES) log-STFT \u2192 conv(8, 3x3) \u2192 pool(2) \u2192 conv(16, 3x3)
\u2192 pool(2) \u2192 flatten \u2192 linear(64) \u2192 linear(4). ~30k parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from structural_listener.model.dataset import SOURCE_TYPES, EventSample
from structural_listener.model.stft import F_BINS, N_FRAMES, stft_input


class AeCnn(nn.Module):
    """Small CNN on the log-STFT spectrogram, 4-class output."""

    def __init__(self, n_classes: int = len(SOURCE_TYPES)) -> None:
        """Build the CNN: two conv blocks + linear head."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # After two 2x pools: F_BINS/4 x N_FRAMES/4 = 8 x 16 (with F_BINS=32, N_FRAMES=64).
        flat = 16 * (F_BINS // 4) * (N_FRAMES // 4)
        self.fc1 = nn.Linear(flat, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, 1, F_BINS, N_FRAMES) \u2192 logits (batch, n_classes)."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits: torch.Tensor = self.fc2(x)
        return logits


def make_stft_batch(samples: list[EventSample]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (X, y) batch of STFT inputs from an EventSample list.

    Uses the strongest detected channel's hit onset to centre the STFT window. Samples
    with no detected hit are skipped.

    Returns (X: float32 (n, 1, F_BINS, N_FRAMES), y: int64 (n,)).
    """
    from structural_listener.dsp.hit_detection import detect_hits

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for s in samples:
        hits = detect_hits(s.waveforms, s.sample_rate_hz)
        if not hits:
            continue
        hit = max(hits, key=lambda h: h.peak_amplitude)
        spec = stft_input(s.waveforms[hit.channel], s.sample_rate_hz, hit.onset_sample)
        xs.append(spec)
        ys.append(s.label)
    if not xs:
        return torch.empty(0, 1, F_BINS, N_FRAMES), torch.empty(0, dtype=torch.int64)
    x = torch.from_numpy(np.stack(xs)[:, None, :, :]).float()
    y = torch.tensor(ys, dtype=torch.int64)
    return x, y


def train_cnn(
    train_samples: list[EventSample],
    test_samples: list[EventSample],
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[AeCnn, dict[str, float | np.ndarray]]:
    """Train the CNN and return (model, report).

    The report dict has: accuracy, confusion (np.ndarray), train_loss (final).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train = make_stft_batch(train_samples)
    x_test, y_test = make_stft_batch(test_samples)
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        raise ValueError("empty STFT batch \u2014 no hits detected in the dataset")

    model = AeCnn()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    final_loss = 0.0
    for _ in range(epochs):
        model.train()
        optim.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optim.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(1)
    acc = float((preds == y_test).float().mean().item())
    n_classes = len(SOURCE_TYPES)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_test.tolist(), preds.tolist(), strict=True):
        confusion[t, p] += 1

    return model, {"accuracy": acc, "confusion": confusion, "train_loss": final_loss}
