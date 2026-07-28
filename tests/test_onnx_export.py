"""Sprint 2.5: ONNX export works and inference produces finite logits.

Uses a tiny CNN (untrained) so the test is fast \u2014 the point is the export pipeline,
not the accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from structural_listener.model.cnn import AeCnn
from structural_listener.model.onnx_export import benchmark_onnx, export_onnx
from structural_listener.model.stft import F_BINS, N_FRAMES


def test_onnx_export_and_inference(tmp_path: Path) -> None:
    """Export a small CNN to ONNX and run inference."""
    model = AeCnn()
    onnx_path = tmp_path / "test_cnn.onnx"
    export_onnx(model, onnx_path)
    assert onnx_path.exists(), "ONNX file not written"

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = np.random.randn(1, 1, F_BINS, N_FRAMES).astype(np.float32)
    out = sess.run(None, {"stft": inp})[0]
    assert out.shape == (1, 4), f"output shape {out.shape}, expected (1, 4)"
    assert np.all(np.isfinite(out)), "output not finite"


def test_benchmark_returns_positive_latency(tmp_path: Path) -> None:
    """benchmark_onnx returns a positive finite latency in ms."""
    model = AeCnn()
    onnx_path = tmp_path / "test_cnn.onnx"
    export_onnx(model, onnx_path)
    latency_ms = benchmark_onnx(onnx_path, n_warmup=2, n_runs=5)
    assert latency_ms > 0.0, f"latency {latency_ms} not positive"
    assert np.isfinite(latency_ms), f"latency {latency_ms} not finite"


def test_cnn_forward_pass_matches_onnx(tmp_path: Path) -> None:
    """The ONNX model and the torch model produce the same logits."""
    model = AeCnn()
    model.eval()
    onnx_path = tmp_path / "test_cnn.onnx"
    export_onnx(model, onnx_path)

    inp = np.random.randn(1, 1, F_BINS, N_FRAMES).astype(np.float32)
    # torch forward
    with torch.no_grad():
        torch_out = model(torch.from_numpy(inp)).numpy()
    # onnx forward
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"stft": inp})[0]
    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-4, atol=1e-5)
