# Structural Listener — Architecture

> Owner: architect. Nothing gets built that isn't described here first.
> Portfolio archetype: **ML engineer / DSP**, with a secondary **TypeScript front-end** story.
> Languages: **Python** (core) + **TypeScript** (UI). Source concept:
> `Innovative Google ADK Kit Uses.md` §2.

## 0. What is simulated, and the correction the source report needs

The source report proposes capturing 10 kHz – 1 MHz acoustic emission on an AIY Voice Kit.
**That is physically impossible** — its MEMS microphones sample at 48 kHz, so Nyquist caps
usable bandwidth at 24 kHz. Real acoustic-emission work needs piezo sensors and a high-rate
DAQ costing thousands. This repo does not pretend otherwise, and the README says so in the
first paragraph. Correcting the premise is a better interview story than quietly ignoring it.

| Layer | Real or simulated |
|-------|-------------------|
| Sensor waveforms | **Simulated** — FDTD Lamb-wave propagation with known source coordinates |
| Wave physics | Real physics, approximated: dispersion, geometric spreading, material attenuation |
| DSP (hit detection, TDOA, STFT) | **Real** |
| Localisation maths | **Real** — closed-form multilateration, no learning involved |
| Classifier | **Real** — trained on simulated data, cross-checked on public recorded audio |
| Reported accuracy on simulated data | **Not scientific evidence.** It validates the pipeline, not the physics. Stated in the README. |

**Why simulate rather than use only public datasets** (this is the question an interviewer
will ask): source localisation requires *known ground-truth source coordinates* from a
*multi-sensor array with known geometry*. Public AE datasets almost never publish both.
Simulation is the only way to get labelled localisation data — so the simulator isn't a
shortcut around missing hardware, it's the only source of the label we need.

## 1. The one demo everything serves

1. UI shows a 1 m × 1 m aluminium plate with four sensors at the corners.
2. `make crack X=0.62 Y=0.31` injects a simulated crack-growth event at that coordinate.
3. Four waveforms arrive at four different times, with different amplitudes. The UI shows
   them live, plus spectrograms.
4. **Classification:** the model labels the event `crack-growth` — distinguishing it from
   `friction`, `impact` and `electrical-noise`.
5. **Localisation:** time-difference-of-arrival across the four sensors is solved by
   least-squares multilateration. A heat map appears on the plate.
6. Predicted position lands within a few centimetres of truth. **The error is printed on
   screen in millimetres.**
7. Inject `impact` at a different coordinate — classified differently, localised correctly.
8. `make replay DATASET=mimii` — the same classifier pipeline runs against **real recorded
   machine audio**, proving the DSP path isn't only a toy.

**The interview line:** *"The localisation is closed-form multilateration and I can derive
it on a whiteboard. The classification is a CNN and I can show you its confusion matrix.
Knowing which of those two problems needed machine learning is most of the work."*

## 2. Component map

```
  sim/  FDTD Lamb-wave solver ──▶ 4× waveform (known source (x,y))
           │
           ▼
  dsp/  hit detection ──▶ TDOA (xcorr + sub-sample interp) ──▶ STFT
           │                        │                            │
           │                        ▼                            ▼
           │              localize/ multilateration      model/ CNN
           │                 least-squares (x̂,ŷ)           4-class
           │                        │                            │
           └────────────────────────┴──────────────┬─────────────┘
                                                   ▼
                                    service/  FastAPI + WebSocket
                                                   │
                                    ui/  Next.js + TS: plate canvas,
                                         live waveforms, spectrograms,
                                         localisation heat map, error readout
```

## 3. The simulator

Deliberately a real piece of physics, not noise with a label.

| Element | Model |
|---------|-------|
| Propagation | 2D FDTD on a thin-plate approximation; A0 and S0 Lamb modes |
| Dispersion | Frequency-dependent group velocity — this is what makes TDOA *hard* and therefore worth demonstrating |
| Attenuation | Geometric spreading (1/√r in a plate) + material damping |
| Reflections | Plate boundaries reflect; a late-arriving echo can defeat naive threshold detection |
| Source signatures | Crack growth = short, broadband, high-frequency burst. Friction = sustained, narrowband. Impact = low-frequency, high-amplitude. Noise = 50 Hz + harmonics. Parameters cited to AE literature in `docs/sources.md`. |
| Sensor model | Piezo response curve, additive Gaussian + 1/f noise, per-channel gain mismatch |

Every parameter is in one config file with its literature source. **If a number can't be
cited, it gets a comment saying it's a guess.**

## 4. Localisation — the part that is not machine learning

Given arrival times `t_i` at sensors `s_i` and wave speed `c`, solve for source `p`:

```
minimise  Σ_i,j  [ (|p - s_i| - |p - s_j|)/c  -  (t_i - t_j) ]²
```

Gauss–Newton from a centroid initial guess. The **residual is reported as a confidence** —
a high residual means the arrival times are geometrically inconsistent, which usually means
a reflection was mistaken for the direct arrival. That failure mode is demonstrated
deliberately in the eval.

Dispersion makes `c` frequency-dependent, so arrival time is estimated at a fixed band. The
error this introduces is measured and reported, not hand-waved.

## 5. The classifier

- Input: log-STFT of a 5 ms window around each detected hit, per channel.
- **Baselines first.** Logistic regression on hand-designed AE features (rise time, ring-down
  count, peak amplitude, energy, duration) — these are the features the AE literature
  actually uses. Then a small CNN.
- Both numbers go in the README. If the hand-feature baseline is within a point of the CNN,
  **say so** — that's a more interesting result than the CNN winning.
- Export to ONNX; benchmark CPU vs. ROCm.

## 6. Repository layout

```
sim/            FDTD solver, source signatures, sensor model, config
dsp/            hit detection, TDOA, STFT, feature extraction
localize/       multilateration solver, residual/confidence
model/          baselines, CNN, training, ONNX export
eval/           localisation error distribution, confusion matrix, failure cases
service/        FastAPI + WebSocket streaming
ui/             Next.js + TypeScript: plate canvas, waveforms, heat map
data/           DVC-tracked: MIMII / recorded-audio slice for the replay demo
docs/           ADRs, sources.md (literature citations), interview stories
```

## 7. Testing

| Layer | What | CI |
|-------|------|----|
| Unit | FDTD solver conserves energy in a lossless configuration | yes |
| Unit | TDOA recovers a known delay from a synthetic pair within tolerance | yes |
| **Numeric** | Multilateration recovers a known source to < 5 cm across a grid of 100 positions | yes |
| **Failure** | A reflection-dominated event produces a high residual, not a confidently wrong answer | yes |
| Model | Confusion matrix floor asserted; regressions fail the build | yes |
| Integration | End-to-end: inject → classify → localise, error under threshold | yes |
| Manual | MIMII replay produces sensible hits | per sprint |

## 8. Known risks

| Risk | Mitigation |
|------|-----------|
| FDTD solver is slow in pure NumPy | Coarse grid sized in Sprint 0 against a wall-clock budget. CuPy/ROCm is an optimisation, never a requirement. |
| Dispersion modelling rabbit-hole | Timeboxed. A single-mode approximation with documented error is acceptable and honest. |
| Classifier trained on simulated data proves nothing | Explicitly acknowledged in the README; the MIMII replay is the cross-check, and it is qualitative — say that. |
| Building a heavy Next.js app eats the schedule | UI is one page. No routing, no auth, no component library beyond a chart lib. |
| Scope creep toward "real SHM system" | No multi-plate, no 3D, no sensor-placement optimisation. Parking lot. |
