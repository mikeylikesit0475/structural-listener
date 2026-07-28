# "Explain it back to me" — Multilateration Derivation (Sprint 3.6)

> The interview prep doc. Derive the localisation objective from first principles,
> cold, without notes. The goal (BACKLOG.md 3.6) is to be able to do this on a
> whiteboard. The answer below is the script.

## Setup

Four sensors at known positions `s_i = (x_i, y_i)`, `i = 0..3`. A source at unknown
position `p = (x, y)` emits a wave that travels at speed `c`. The wave arrives at
sensor `i` at time `t_i`. We don't know the emission time `t_0`; we only know the
arrival times (or, more precisely, the *differences* of arrival times — a single
clock offset is unobservable).

## Step 1: The forward model

The travel time from `p` to `s_i` is the Euclidean distance divided by the wave speed:

```
t_i = t_0 + |p - s_i| / c
```

where `|p - s_i| = sqrt((x - x_i)^2 + (y - y_i)^2)`.

## Step 2: Eliminate the unknown emission time

We don't know `t_0`, and we don't need it. Subtract the reference sensor (say `i=0`)
from every other sensor:

```
t_i - t_0 = (|p - s_i| - |p - s_0|) / c
```

The left side is the **time-difference of arrival (TDOA)** — a measured quantity.
The right side is the **difference of geometric distances**, divided by `c`. Call
the measured TDOA `τ_i = t_i - t_0`. Then:

```
|p - s_i| - |p - s_0| = c · τ_i        for i = 1, 2, 3
```

This is one equation per sensor (excluding the reference). Three equations, two
unknowns (`x, y`) — an overdetermined system. That's why four sensors is the
minimum for 2D multilateration (three TDOA equations from four sensors).

## Step 3: The objective (least squares)

The three (or more) TDOA equations are not exactly satisfiable because the
measured `τ_i` have noise. So we minimise the squared residual:

```
minimise_p  Σ_i  [ (|p - s_i| - |p - s_0|) / c  -  τ_i ]^2
```

This is the objective in `ARCHITECTURE.md §4` and `localize/multilateration.py`.
It's a sum of squared nonlinear residuals — a nonlinear least-squares problem.

## Step 4: Gauss–Newton

Let `r_i(p) = (|p - s_i| - |p - s_0|) / c - τ_i` be the residual for sensor `i`.
Stack them into a vector `r(p)`. We want `Σ r_i^2 = r^T r` minimised.

The gradient is `2 J^T r`, where `J` is the Jacobian `∂r/∂p`. The Jacobian entry
for sensor `i`, position component `k` is:

```
∂r_i/∂p_k = (1/c) · [ (p_k - s_{i,k}) / |p - s_i|  -  (p_k - s_{0,k}) / |p - s_0| ]
```

That's `(1/c)` times the difference of unit vectors from each sensor to the
source. Geometric: the residual changes fastest when you move `p` along the
direction that changes the distance difference most.

Gauss–Newton update: solve `J^T J δ = -J^T r`, then `p ← p + δ`. Repeat until
`|δ| < tolerance`. We add damping (`δ ← 0.5 · δ`) for stability with poor initial
guesses, and start from the centroid of the sensors.

## Step 5: The residual is the confidence

At convergence, the residual `r^T r / n` (RMS) is the **confidence score**. If the
measured TDOAs are geometrically consistent (a real source exists at the estimated
position), the residual is small. If a reflection was mistaken for the direct
arrival at one sensor, that TDOA is inconsistent with the others, and the residual
spikes. We map residual → confidence as `1 / (1 + residual / 50µs)`, so a 50µs
residual gives 50% confidence.

This is ADR-006: report uncertainty, not a confidently wrong coordinate.

## What I'd say on the whiteboard

> "Four sensors at known positions. Source emits at unknown time. Arrival time at
> sensor i is travel distance over wave speed, plus the emission time. Subtract the
> reference sensor to kill the emission time — you're left with TDOA equals distance
> difference over speed. Three of those equations from four sensors, two unknowns,
> so it's overdetermined. Minimise the squared residual by Gauss–Newton. The
> residual at convergence is the confidence: a reflection corrupts one TDOA, the
> residual spikes, and the system reports low confidence instead of a wrong
> coordinate."