"""ONNX export and CPU/ROCm benchmark for the CNN (Sprint 2.5).

Exports the trained CNN to ONNX, then benchmarks inference latency on CPU and (if
available) ROCm. Per CLAUDE.md, ROCm on gfx1151 is Preview tier with an open GPU-hang
issue under mixed load, so the ROCm benchmark is wrapped in a try/except \u2014 if it
fails or falls back to CPU, that's the honest result and we say so.

The CPU number is the one that matters for the demo (the Sprint 3 UI runs on CPU).
Both numbers go in the README.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from structural_listener.model.cnn import AeCnn, train_cnn
from structural_listener.model.dataset import generate_dataset
from structural_listener.model.stft import F_BINS, N_FRAMES


def export_onnx(model: AeCnn, out_path: Path) -> None:
    """Export the CNN to ONNX. Input: (1, 1, F_BINS, N_FRAMES) float32."""
    model.eval()
    dummy = torch.zeros(1, 1, F_BINS, N_FRAMES)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["stft"],
        output_names=["logits"],
        dynamic_axes={"stft": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def benchmark_onnx(
    onnx_path: Path, n_warmup: int = 5, n_runs: int = 50, batch_size: int = 1
) -> float:
    """Benchmark ONNX inference latency. Returns median latency in milliseconds."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = np.random.randn(batch_size, 1, F_BINS, N_FRAMES).astype(np.float32)
    for _ in range(n_warmup):
        sess.run(None, {"stft": inp})
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"stft": inp})
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def main() -> int:
    """Train, export, benchmark. Prints both numbers for the README."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("generating training dataset (50/class for speed)...")
        train = generate_dataset(n_per_class=50, out_dir=tmp_path, seed=1)
        print("generating test dataset (20/class)...")
        test = generate_dataset(n_per_class=20, out_dir=tmp_path, seed=99)

    model, report = train_cnn(train, test, epochs=20)
    print(f"CNN trained: accuracy = {float(report['accuracy']):.1%}")

    onnx_path = Path("data/cnn.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, onnx_path)
    print(f"exported ONNX: {onnx_path}")

    cpu_ms = benchmark_onnx(onnx_path)
    print(f"CPU inference (median over 50 runs): {cpu_ms:.2f} ms")

    # ROCm benchmark: per CLAUDE.md, gfx1151 is Preview tier. onnxruntime may not have
    # a ROCmExecutionProvider build available, or it may fall back to CPU silently.
    # We check the loaded providers explicitly and report honestly.
    try:
        sess_rocm = ort.InferenceSession(str(onnx_path), providers=["ROCmExecutionProvider"])
        loaded = sess_rocm.get_providers()
        if "ROCmExecutionProvider" not in loaded:
            print(f"ROCm: not available (loaded providers: {loaded}). CPU only.")
            print("  (CLAUDE.md: gfx1151 is Preview tier; this is the expected result.)")
        else:
            inp = np.random.randn(1, 1, F_BINS, N_FRAMES).astype(np.float32)
            for _ in range(5):
                sess_rocm.run(None, {"stft": inp})
            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                sess_rocm.run(None, {"stft": inp})
                times.append((time.perf_counter() - t0) * 1e3)
            rocm_ms = float(np.median(times))
            print(f"ROCm inference (median over 50 runs): {rocm_ms:.2f} ms")
    except Exception as e:
        print(f"ROCm: unavailable on this machine ({type(e).__name__}). CPU only.")
        print("  (CLAUDE.md: gfx1151 is Preview tier; this is the expected result.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
