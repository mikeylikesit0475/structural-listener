# Structural Listener — Backlog & Sprint Plan

Scrum master's file. **If an item blows its box it gets cut or split, not extended.**

## Definition of done (every sprint)

- [ ] Working demo of the sprint goal, no code edits
- [ ] Tests green in CI
- [ ] `DEMO_SCRIPT.md` updated to match reality
- [ ] One interview story in `docs/interview-stories/`

## Definition of done (whole repo)

README leading with **the simulation boundary statement**, then localisation error
distribution and confusion matrix · demo GIF · architecture sketch · tests · CI ·
Dockerfile · 2-minute live demo.

---

## Sprint 0 — Physics · goal: *"four sensors receive a wave from a known source"*

No ML. Get the simulator honest and fast enough first — everything downstream is built on
its output.

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 0.1 | Repo init, `uv`/`pyproject`, ruff + mypy, CI, push | 2h | Green CI |
| 0.2 | FDTD solver, 2D thin plate, single mode; **wall-clock budget: one event < 5 s** | 6h | Energy-conservation test passes; grid size chosen against the budget |
| 0.3 | Source signatures: crack / friction / impact / noise, parameters cited in `docs/sources.md` | 4h | Four visibly distinct waveforms |
| 0.4 | Sensor model: attenuation, per-channel noise and gain mismatch | 3h | Four channels differ realistically |
| 0.5 | Dispersion: frequency-dependent group velocity, error documented | 4h | Measured TDOA bias recorded in `docs/` |
| 0.6 | CLI: `make crack X= Y=` writes a 4-channel `.npz` with ground truth | 2h | File contains waveforms + true coordinates |

**Demo:** inject an event, plot four waveforms, point out the arrival-time differences.
**Total ~21h.** 0.2 is the overrun risk — if it blows the box, coarsen the grid and record
the resolution cost rather than optimising.

---

## Sprint 1 — Localisation · goal: *"it finds the crack, and I can derive why"*

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 1.1 | Hit detection: adaptive threshold, duration gate, rise-time measurement | 4h | Detects all four event types reliably |
| 1.2 | TDOA by cross-correlation with sub-sample (parabolic) interpolation | 4h | Recovers a known synthetic delay within tolerance |
| 1.3 | Gauss–Newton multilateration + residual as confidence | 5h | **Recovers a known source to < 5 cm across a 100-position grid** — asserted in CI |
| 1.4 | Error distribution study: accuracy vs. position, incl. near-edge degradation | 3h | Heat map of error committed to `docs/` |
| 1.5 | **Failure case:** reflection-dominated event → high residual, not a confident wrong answer | 3h | Test proves the system reports low confidence |

**Demo:** inject at an arbitrary coordinate, get the position back with the error in
millimetres, then inject near an edge and show the confidence drop. **Total ~19h.**

---

## Sprint 2 — Classification · goal: *"and it knows what kind of event it was"*

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 2.1 | Classical AE features: rise time, ring-down count, energy, duration, peak | 3h | Feature extractor tested |
| 2.2 | **Baseline first:** logistic regression + random forest on those features | 3h | Confusion matrix recorded. This number goes in the README regardless of what follows |
| 2.3 | STFT front-end + small CNN in PyTorch | 5h | Trains; both numbers side by side |
| 2.4 | Honest comparison written up — including the possibility the baseline wins | 2h | `docs/interview-stories/baseline-vs-cnn.md` |
| 2.5 | ONNX export + CPU vs. ROCm benchmark | 3h | Both latencies in the README |
| 2.6 | Confusion-matrix floor asserted in CI | 2h | Regression fails the build |
| 2.7 | `make replay DATASET=mimii`: same DSP path over real recorded audio | 4h | Produces sensible hits; **labelled qualitative, not scored** |

**Demo:** four event types classified, confusion matrix on screen, then the real-audio
replay. **Total ~22h.**

---

## Sprint 3 — Surface & polish · goal: *"interview-ready"*

| # | Item | Box | Done when |
|---|------|-----|-----------|
| 3.1 | FastAPI + WebSocket: stream waveforms, hits, verdicts | 4h | Client receives a live event |
| 3.2 | Next.js + TS: plate canvas, sensor markers, live waveforms | 6h | One page, no routing |
| 3.3 | Spectrogram view + localisation heat map + **error readout in mm** | 5h | Error is visible on screen during the demo |
| 3.4 | Dockerfile + compose | 2h | `docker compose up` runs it |
| 3.5 | README: simulation boundary first, then the numbers | 3h | A stranger knows what's real in 30 seconds |
| 3.6 | **"Explain it back to me"** — derive the multilateration objective on paper, cold | 2h | Done without notes |
| 3.7 | Demo GIF + finalise `DEMO_SCRIPT.md` | 3h | Matches reality |

**Total ~25h.** Program total ≈ **87h**.

---

## Parking lot

- 3D localisation / plate thickness
- Multi-mode dispersion (A0 + S0 jointly)
- Sensor-placement optimisation
- Real piezo hardware + a cheap high-rate DAQ
- Kalman tracking of a propagating crack front
