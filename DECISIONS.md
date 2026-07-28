# Architecture Decision Records — Structural Listener

Format: context → decision → consequence.

---

## ADR-001 — Correct the source report's premise in public

**Context.** The concept note proposes 10 kHz – 1 MHz acoustic emission capture on AIY
Voice Kit hardware. Its MEMS microphones sample at 48 kHz; Nyquist caps usable bandwidth at
24 kHz. The proposal is not merely impractical, it is impossible.

**Decision.** State this in the README's first paragraph and in `docs/sources.md`, then
build the thing that *is* possible.

**Consequence.** The repo opens by demonstrating that its author checks physical
feasibility before writing code. An interviewer who knows AE will recognise the correction
immediately; one who doesn't will still see the reasoning. Either way it beats a README
that quietly drops an impossible claim.

---

## ADR-002 — Simulate, because the label we need doesn't exist in public data

**Context.** Public AE and machine-sound datasets exist (MIMII, NASA IMS, various Zenodo
sets). Using only those would avoid writing a solver.

**Decision.** Write the FDTD simulator anyway.

**Consequence.** Source localisation needs known ground-truth source coordinates from a
multi-sensor array with known geometry. Public datasets essentially never publish both. The
simulator is not a workaround for absent hardware — it is the only source of the label the
project's central capability requires. That is the answer to "why not just use real data,"
and it is a real answer.

Cost: the classifier's accuracy on simulated data proves nothing about real structures. Two
mitigations, both honest rather than clever: the README says so, and the MIMII replay
demonstrates the DSP path over real recordings as a **qualitative** cross-check.

---

## ADR-003 — Localisation is not a machine-learning problem

**Context.** A neural network could regress source position from waveforms directly.

**Decision.** Closed-form multilateration by Gauss–Newton least squares.

**Consequence.** The solution is derivable on a whiteboard, needs no training data, has an
interpretable residual that doubles as a confidence score, and degrades in a way we can
explain (geometrically inconsistent arrivals). A regression network would be more
impressive-sounding and strictly worse on all four counts. **Knowing which half of a system
needs ML is the skill this repo is trying to evidence.**

---

## ADR-004 — Classical baselines before the CNN, and both numbers published

**Context.** The AE literature uses hand-designed features — rise time, ring-down count,
energy, duration. A CNN on spectrograms is the modern default.

**Decision.** Implement the classical baseline first, record its confusion matrix, then
build the CNN. Publish both regardless of which wins.

**Consequence.** If the baseline is within a point of the CNN, that is the more interesting
result and the README will say so. Most portfolio repos show only the deep model, which
tells a reader nothing about whether it was needed. This costs one extra sprint item and
buys the ability to answer "did you need the neural network?" with a number.

---

## ADR-005 — Every simulator constant is cited or flagged as a guess

**Context.** A physics simulator is only as credible as its parameters. It is easy to
invent plausible numbers.

**Decision.** One config file. Every value carries either a literature citation in
`docs/sources.md` or an inline comment saying it is a guess.

**Consequence.** An interviewer can audit the physics. More usefully, it forces honesty
during development — the discipline of writing "guess" next to a number tends to prompt
looking the number up.

---

## ADR-006 — The reflection failure mode is demonstrated, not hidden

**Context.** Plate boundaries reflect. A reflected arrival can be mistaken for the direct
one, producing a confidently wrong position.

**Decision.** Include this case in the eval and in the demo. High residual → low confidence
→ the system reports uncertainty rather than a wrong coordinate. There is a test.

**Consequence.** The demo includes a step where the system fails gracefully. That is worth
more than a demo where everything works, because it is the question a sceptical engineer
was already going to ask.

---

## ADR-007 — One page of TypeScript, not an application

**Context.** The portfolio wants a TS signal from this repo, and a rich UI is tempting.

**Decision.** Next.js, a single page, no routing, no auth, no component library beyond a
charting dependency. The TS story here is real-time WebSocket streaming and canvas
rendering, which is enough.

**Consequence.** The UI cannot eat the schedule. If a TS-heavy role needs a deeper artifact,
`veritas-engine`'s React Flow graph or the Track 2 embeddable-widget project carries it —
this repo is an ML/DSP repo with a good front end, and it should read that way.
