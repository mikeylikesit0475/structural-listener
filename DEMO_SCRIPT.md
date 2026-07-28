# Structural Listener — 2-Minute Demo Script

Live, no code edits. **Never describe a capability that doesn't work** — remove the step
instead.

## Pre-flight

```bash
docker compose up -d
open http://localhost:3000
```

Have `docs/figures/error-heatmap.png` and the confusion matrix open in tabs. Warm the
FDTD cache so step 2 doesn't stall. A demo GIF is at `docs/figures/demo.gif`
(generate with a screen recorder if missing — the live demo is the point).

---

## The script

**[0:00] Lead with the correction. It sets the tone.**

> "The brief for this was 'acoustic-emission monitoring on a cheap Raspberry Pi voice kit.'
> That's physically impossible — those microphones sample at 48 kHz, so you top out at
> 24 kHz, and crack emissions live between about 100 kHz and a megahertz. So I built the
> part that's real: the physics simulator and the full signal chain."

Point at the plate on screen. Four sensors, one per corner.

**[0:20] Inject an event.**

In the UI, set type=crack, x=0.62, y=0.31, click "inject".

> "That's a simulated crack-growth event. Four sensors, four different arrival times,
> different amplitudes — the wave attenuates and disperses on the way."

Four waveforms populate. Point at the arrival-time offsets.

**[0:40] Classification.**

> "First question is what kind of event. Crack growth, friction, impact, or electrical
> noise. That's the CNN."

Show the label. Then flip to the confusion matrix tab briefly.

> "And here's the thing I'd want you to push on — the classical baseline, using the
> rise-time and ring-down features the AE literature actually uses, gets **99.5%**.
> The CNN gets **76%**. Both are in the README. I built the baseline first on purpose,
> and the baseline wins — the CNN confuses crack with noise because the grid filters
> the 80 kHz carrier. That's the more interesting result."

**[1:00] Localisation. Slow down — this is the technical centrepiece.**

> "Second question is where. That one isn't machine learning at all. It's the time
> differences between the four arrivals, solved by least squares — multilateration. I can
> derive it on a whiteboard."

Heat map appears on the plate.

> "Predicted position, true position, and the error in millimetres." **Read the number out.**
> Median error across a 100-position grid is 2.8 cm; the error heat map is in `docs/figures/`.

**[1:20] The failure case. Do not skip this.**

Inject an event near a plate edge.

In the UI, set type=crack, x=0.05, y=0.92, click "inject".

> "Near an edge, a boundary reflection can arrive before you'd expect and get mistaken for
> the direct wave. Watch the confidence."

Residual spikes, confidence drops.

> "The solver's residual is the confidence. Geometrically inconsistent arrival times mean
> it says 'I don't know' instead of giving you a confident wrong coordinate. There's a test
> for that case."

**[1:40] The real-data cross-check.**

Place a wav file (e.g. a MIMII slice) under `data/replay/`, then:

```bash
make replay
```

> "Same DSP path, real recorded machine audio from a public dataset. This is qualitative —
> I'm not claiming a score on it. It's there to show the signal chain works on something I
> didn't generate."

**[1:55] Close on the boundary.**

> "To be explicit about what's real: the physics, the DSP, the multilateration and the
> model are real. The waveforms are simulated, so the accuracy number validates the
> pipeline, not the structural engineering. That's in the first paragraph of the README."

---

## Questions to have answers ready for

| Question | Answer lives in |
|----------|-----------------|
| "Why not use a real AE dataset?" | ADR-002 — public sets don't publish source coordinates *and* array geometry. Best question they can ask; have this crisp. |
| "Why isn't localisation a neural net?" | ADR-003. Derivable, no training data, interpretable confidence. |
| "Did you need the CNN?" | The two numbers. If the baseline is close, say so — it's the stronger answer. |
| "Where do your simulator constants come from?" | `docs/sources.md`. Everything is cited or flagged as a guess. |
| "What would make this real?" | Piezo sensors + a high-rate DAQ, ~£2–4k. Parking lot has the plan. |
