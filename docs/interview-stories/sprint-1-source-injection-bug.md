# Interview Story — Sprint 1.3: The Source That Never Moved

> The headline a candidate can actually use in an interview: *"I had a localisation
> pipeline stuck at ~54 cm median error for hours. A second pass found a misplaced
> parenthesis that pinned every simulated source at the bottom-left corner of the
> plate. The clue was that two sensors on opposite corners produced identical arrival
> picks — only physically possible if the source lies on the plate's diagonal."*

## The setup

Sprint 1.3 was Gauss–Newton multilateration. The solver maths was easy to verify in
isolation: with noise-free synthetic arrival times, it recovers the source to < 1 mm.
The hard part was the end-to-end pipeline — `inject → TDOA → multilateration` — on a
real (simulated) reflective plate.

I ran a 100-position grid and got a median error of ~54 cm. That number is suspicious
on its own: it's close to the average distance from a grid point to a corner of a 1 m
plate. I didn't notice that at the time. I went down the path every AE engineer would:
the TDOA must be catching reflections.

I tried, in rough order:
- Stronger absorbing-boundary sponge (reflections attenuate more). Got *worse* — 96 cm.
  The sponge killed the direct arrival at the far sensors.
- STA/LTA first-arrival picking (the seismology standard). 72 cm. The band-limited noise
  has correlated bursts that fire the threshold before the real wave.
- A `c_eff` sweep. The median dropped to ~29 cm at `c_eff=1100` on a small grid, then
  bounced back to 72 cm on the full grid. Inconsistent.
- A larger plate (2 m), to separate the direct arrival from reflections in time. No
  improvement.

Hours in, the honest conclusion looked like: the 5 cm target is over-ambitious for a
reflective 1 m plate with 4 sensors and a dispersive mode. I committed at ~54 cm with
a residual-as-confidence mitigation and a note that the TDOA was the gap.

## The clue a second pass caught

Channels 1 and 3 — sensors at opposite corners `(0.95, 0.05)` and `(0.05, 0.95)` —
were producing arrival picks that agreed to within 0.1 µs. That's not a reflection
problem. That's not a dispersion problem. That's only physically possible if the
wavefield is symmetric about the plate's diagonal, i.e. the source lies *on* it.

If the source lies on the diagonal for *every* requested `(x, y)`, then the source
position isn't being honoured at all.

## The bug

In `sim/signatures.py:38` and `sim/sources.py:56`:

```python
self._i = min(max(round(source_xy_m[0] / dx), 1, nx - 2), 1)  # buggy
self._i = min(max(round(source_xy_m[0] / dx), 1), nx - 2)  # fixed
```

The misplaced parenthesis makes `max(v, 1, nx-2)` always return `nx - 2`, then
`min(…, 1)` always returns `1`. So every event physically originated at grid cell
`(1, 1)` — the bottom-left corner — while the `.npz` ground truth recorded whatever
`(x, y)` the caller requested.

The localisation solver was working correctly. It was solving for the source that
actually existed — the one pinned at the corner — and the "error" was the distance
from the requested coordinate to the corner. ~54 cm is exactly the average
distance from a 10×10 interior grid to `(0, 0)`.

Every symptom I'd chased for hours was a consequence:
- The "boundary artifacts at corner sensor 0": the source was 6 cm away from it.
- The `c_eff` sweep finding `~1100` as a sweet spot: the geometry was fixed, so I was
  fitting a single constant to a corner-anchored system, not a real wave-speed
  calibration.
- The STA/LTA "noise triggers" on the near sensor: they weren't noise. The source was
  next to that sensor.

## The fix and the new number

The fix is one parenthesis. With it, the gated envelope-peak picker on the 20–90 kHz
band, and `c_eff = 1200 m/s` (the grid group velocity — a *cited* constant, see
`docs/sources.md`, not a calibration), the 100-position grid gives:

```
median = 2.8 cm,  mean = 3.2 cm,  p90 = 6.2 cm,  max = 11.3 cm
```

The Sprint 1 target (< 5 cm median) is met. Two supporting corrections:
- `theoretical_group_velocity` was missing the factor of 2 (it computed phase
  velocity; for the Kirchhoff plate `c_g = 2 c_p`). Fixed.
- A regression test (`tests/test_source_injection.py`) now asserts the injected grid
  cell matches the requested coordinates and that two different positions produce
  different waveforms.

## Why this is the interview story

Three things, in order of importance:

1. **The symmetry argument was the diagnostic.** I had a 100-position error
   distribution, a `c_eff` sweep, four different TDOA methods, and hours of tuning.
   None of it pointed at the bug. The thing that pointed at the bug was a one-line
   symmetry observation: two channels on opposite corners can't agree unless the
   source is on the diagonal. That's the kind of physics-first reasoning that beats
   parameter sweeps.

2. **The bug was in the simulator, not the DSP.** Every fix I tried was in the TDOA.
   The DSP was correct the whole time. This is the argument for testing each layer in
   isolation *and* having a ground-truth invariant (here: the injected cell matches
   the requested coordinates) that crosses layer boundaries.

3. **The constant that "fixed" it was a citation, not a calibration.** After the bug
   fix, `c_eff = 1200 m/s` falls out of the grid group-velocity cap
   `2·c_basis/dx ≈ 1237 m/s`. A finer grid would raise the cap and the continuum c_g
   would take over. The number goes in `docs/sources.md`, not a magic constant in a
   test.

## What I'd say if asked "tell me about a bug that took you a long time"

> "I had a localisation pipeline stuck at ~54 cm median error for hours. I tried four
> different TDOA methods, a wave-speed sweep, a larger plate — none of it helped. The
> clue that broke it was that two sensors on opposite corners of the plate produced
> identical arrival times. That's only physically possible if the source lies on the
> plate's diagonal for every event. It did: a misplaced parenthesis in the source
> constructor pinned every event at the bottom-left corner of the grid, while the
> ground-truth file recorded the requested coordinates. The solver was correct the
> whole time; it was localising the source that actually existed. After the one-line
> fix, the median error dropped to 2.8 cm — under the 5 cm target. The lesson was to
> check the layer boundary first: the simulator and the DSP were both correct in
> isolation, but the contract between them (the source position) was broken, and no
> amount of DSP tuning could fix that."