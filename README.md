# Structural Listener

**The waveforms in this repo are simulated, not recorded.** An FDTD thin-plate solver
generates them with known source coordinates. The DSP, localisation, and classifier
are real; the accuracy figures validate the pipeline, not the physics. A real
acoustic-emission system needs piezo sensors and a high-rate DAQ — the README's first
job is to say that clearly, because the source report that prompted this project
proposed the impossible (capturing 100 kHz–1 MHz AE on a 48 kHz MEMS microphone).

---

## What it does

A 1 m × 1 m aluminium plate. Four sensors at the corners. Inject a simulated
acoustic-emission event — crack growth, friction, impact, or electrical noise — and
the system:

1. **Classifies** the event type (CNN + classical baseline, both numbers below).
2. **Localises** the source by closed-form multilateration (no machine learning).
3. **Reports a confidence** — the multilateration residual, which spikes when a
   boundary reflection is mistaken for the direct arrival.

The error is printed on screen in millimetres. There's a deliberate failure case
in the demo: inject near an edge, watch the confidence drop.

## The numbers

### Localisation (Sprint 1)

100-position grid, gated envelope-peak TDOA + Gauss–Newton multilateration,
`c_eff = 1200 m/s` (grid group velocity, cited in `docs/sources.md`):

| Statistic | Value |
|-----------|-------|
| Median error | **2.8 cm** |
| Mean error | 3.2 cm |
| 90th percentile | 6.2 cm |
| Max error | 11.3 cm |

Heat map: `docs/figures/error-heatmap.png`. The edges degrade — that's the ADR-006
reflection failure mode, demonstrated not hidden.

### Classification (Sprint 2)

Test set: 50 events per class (200 total), held out from 100/class training.

| Model | Accuracy | Note |
|-------|----------|------|
| **Logistic regression** (5 AE features) | **99.5%** | Baseline wins |
| **Random forest** (5 AE features) | **99.5%** | Baseline wins |
| CNN (STFT, 30k params, 30 epochs) | 76.0% | Confuses crack ↔ noise |

The classical features beat the CNN. The reason is in the grid: at dx = 5 mm the
crack source's 80 kHz carrier is numerically filtered, so crack and the 50 Hz noise
both look low-frequency in the STFT. The 5 classical features (rise time, ring-down
count, peak amplitude, energy, duration) separate them by peak amplitude (10×
difference) and duration. **Knowing which half of a system needs ML is the skill this
repo is trying to evidence** (ADR-003, ADR-004).

The honest comparison writeup: `docs/interview-stories/baseline-vs-cnn.md`.

### Inference latency

| Backend | Latency |
|---------|---------|
| CPU (ONNX Runtime) | **0.15 ms** |
| ROCm | Unavailable on this machine (gfx1151 Preview tier; see CLAUDE.md) |

## Architecture

```
  sim/  FDTD Lamb-wave solver ──▶ 4× waveform (known source (x,y))
           │
           ▼
  dsp/  hit detection ──▶ TDOA (gated envelope-peak) ──▶ STFT
           │                        │                            │
           │                        ▼                            ▼
           │              localize/ multilateration      model/ CNN + baseline
           │                 least-squares (x̂,ŷ)           4-class
           │                        │                            │
           └────────────────────────┴──────────────┬─────────────┘
                                                   ▼
                                    service/  FastAPI + WebSocket
                                                   │
                                    ui/  Next.js + TS: plate canvas,
                                         live waveforms, spectrogram,
                                         localisation heat map, error in mm
```

- **Localisation** is closed-form Gauss–Newton multilateration. Derivable on a
  whiteboard. The residual is the confidence score. No ML.
- **Classification** has both a 5-feature logistic regression baseline and a CNN.
  Both numbers are in the README regardless of which wins (ADR-004).
- **Every simulator constant** is cited in `docs/sources.md` or flagged as a guess
  (ADR-005).

## Quick start

```bash
# Install
make install

# Run the gates
make check   # ruff + mypy --strict + pytest

# Inject a simulated event
make crack X=0.62 Y=0.31

# Run the service + UI (two terminals)
make serve                    # FastAPI on :8000
cd ui && npm install && npm run dev   # Next.js on :3000

# Or with docker
docker compose up -d          # :3000 (UI) + :8000 (API)

# Regenerate the error heat map
make heatmap

# Replay real recorded audio (qualitative, not scored)
# Place wav files under data/replay/ then:
make replay
```

## Tests

`uv run pytest` — 47+ tests including:
- FDTD energy conservation (Sprint 0)
- TDOA recovers a known synthetic delay within tolerance (Sprint 1.2)
- **Multilateration recovers a known source to < 5 cm median over 100 positions** (Sprint 1.3, CI-asserted)
- Reflection failure case: high residual → low confidence (Sprint 1.5, ADR-006)
- Source-injection regression: the requested coordinates match the injected cell (Sprint 1.3)
- Classification confusion-matrix floor: 60% accuracy + per-class recall ≥ 50% (Sprint 2.6)
- ONNX export + torch↔onnx parity (Sprint 2.5)
- FastAPI service + WebSocket stream (Sprint 3.1)

Gates: `ruff` + `mypy --strict`.

## The interview line

> "The localisation is closed-form multilateration and I can derive it on a whiteboard.
> The classification is a CNN and I can show you its confusion matrix. Knowing which of
> those two problems needed machine learning is most of the work."

## What's real, what's simulated

| Layer | Real or simulated |
|-------|--------------------|
| Sensor waveforms | **Simulated** — FDTD Lamb-wave propagation, known source coords |
| Wave physics | Real physics, approximated: dispersion, geometric spreading, material attenuation |
| DSP (hit detection, TDOA, STFT) | **Real** |
| Localisation maths | **Real** — closed-form multilateration, no learning |
| Classifier | **Real** — trained on simulated data, cross-checked on real audio (qualitative) |
| Accuracy on simulated data | **Not scientific evidence.** Validates the pipeline, not the physics. |

## Repo layout

```
sim/            FDTD solver, source signatures, sensor model, config
dsp/            hit detection, TDOA, feature extraction
localize/       multilateration solver, residual/confidence
model/          baselines, CNN, training, ONNX export, replay, features
eval/           localisation error distribution, heat map
service/        FastAPI + WebSocket streaming
ui/             Next.js + TypeScript: plate canvas, waveforms, heat map
docs/           ADRs, sources.md, interview stories, figures
data/           generated .npz files, ONNX model, replay audio
```

## Decisions

See `DECISIONS.md` for the ADRs. The key ones:
- **ADR-001:** Correct the source report's impossible premise in public.
- **ADR-003:** Localisation is not ML. Closed-form multilateration.
- **ADR-004:** Baseline before CNN, both numbers published. Baseline wins.
- **ADR-006:** Reflection failure case stays in the demo.
- **ADR-007:** One page of TypeScript, not an application.