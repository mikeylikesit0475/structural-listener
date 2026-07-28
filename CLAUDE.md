# Working agreement for AI assistants in this repo

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) before writing code. It is the contract.
[`DECISIONS.md`](DECISIONS.md) explains *why*. Contradict an ADR only by superseding it in
writing.

## Non-negotiables

1. **This is a portfolio proof-of-concept, not a structural-health-monitoring system.** No
   3D, no multi-plate, no sensor-placement optimisation. If it doesn't improve the
   2-minute demo, cut it.
2. **The simulation boundary is stated in the README's first paragraph** and must stay
   there. Accuracy on simulated data validates the pipeline, not the physics. Do not soften
   this into marketing language.
3. **Localisation stays closed-form.** Do not replace multilateration with a regression
   network. See ADR-003 — this is the repo's central technical argument.
4. **The classical baseline ships alongside the CNN**, with both numbers in the README,
   even — especially — if the baseline wins. See ADR-004.
5. **Every simulator constant is cited in `docs/sources.md` or commented as a guess.**
   See ADR-005.
6. **The reflection failure case stays in the eval and the demo.** ADR-006.
7. **The UI is one page.** No routing, no auth, no design system. ADR-007.

## Conventions

- The FDTD solver has an energy-conservation test. Any change to it must keep that passing.
- Numeric assertions use explicit tolerances with a comment justifying the value — never a
  bare `approx`.
- Ground truth travels with simulated data (`.npz` contains the true source coordinates).
  Never infer ground truth downstream.
- Real recorded data (`make replay`) is **qualitative only**. Do not compute or publish an
  accuracy figure on it — the labels aren't comparable.
- `ruff` + `mypy --strict` are gates.

## Environment notes for this machine

- Dev box is a Ryzen AI Max+ 395 on Fedora. **ROCm on gfx1151 is Preview tier**, not in the
  production support matrix, with an open GPU-hang issue under mixed load. CuPy/ROCm
  acceleration of the FDTD solver is an **optimisation, never a requirement** — NumPy CPU
  must remain a working path.
- **The XDNA2 NPU is unusable on Linux** (Vitis AI EP userspace wheel unpublished for
  x86_64 Linux; ONNX Runtime silently falls back to CPU). Do not write NPU code, and do not
  report an "NPU" benchmark number.

## Working rhythm

Sprints and timeboxes are in `BACKLOG.md`. Sprint 0.2 (the solver) has a wall-clock budget —
if it blows the box, **coarsen the grid and record the resolution cost** rather than
optimising. **Say when an item is going to overrun; don't absorb it.** Each sprint ends with
an interview story in `docs/interview-stories/`.
