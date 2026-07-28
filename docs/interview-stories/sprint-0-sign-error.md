# Interview Story \u2014 Sprint 0: The Sign Error the Energy Test Caught

> The headline a candidate can actually use in an interview: *"I wrote a physics solver
> and a conservation test for it. The test caught a sign error in the equation of motion
> before I ever looked at a waveform. The solver would have produced plausible-looking
> output that was exponentially diverging."*

## The setup

Sprint 0 was the FDTD thin-plate solver. The Kirchhoff plate equation is

  rho h * d\u00b2w/dt\u00b2 + D * \u2207\u2074 w = s(x, y, t)

with auxiliary-field formulation `phi = \u2207\u00b2 w`, giving two coupled Laplacians and
a leapfrog time step. I picked the grid (dx = 5 mm) against a wall-clock budget (< 5 s
per event) by benchmarking NumPy throughput before writing the solver \u2014 5 ms of
physics at N=200 takes ~370 ms, comfortably inside the box.

I also wrote the energy-conservation test *before* I had a working source signature. The
plan was: inject a short pulse, let the field evolve freely, and assert the total
mechanical energy (kinetic + bending) drifts by less than 5% over 2 ms. Leapfrog is
symplectic; for a lossless plate with clamped boundaries, that's the invariant.

## What happened

The test blew up. Overflow warnings, then NaNs, within ~22 steps.

I went down the wrong trail first. I assumed the CFL condition was wrong \u2014 the
continuous-formula bound `dt \u2264 dx\u00b2/(4\u221a(D/\u03c1h))` is what I'd coded. I checked it
numerically: power iteration on the discrete biharmonic operator gave a max eigenvalue
matching the continuous prediction to 0.3%, and my dt was at 0.6\u00d7 the stable bound.
CFL was correct.

Then I tested with and without the clamped-boundary zeroing \u2014 same blow-up. The
boundary wasn't it.

The leapfrog amplification polynomial for `ddot{w} = -\u03bb w` is `g\u00b2 - (2 - dt\u00b2\u03bb)g + 1 = 0`,
stable iff `dt\u00b2\u03bb \u2264 4`. I confirmed `dt\u00b2 \u00b7 \u03bb_max = 1.44`, well inside [0, 4]. The
highest mode was *provably* stable.

## The bug

The equation I'd coded was `ddot{w} = +(D/\u03c1h) \u00b7 lap(lap(w))`.

The discrete Laplacian is negative-semidefinite, so `lap(lap(w))` is
positive-semidefinite. The plus sign makes `ddot{w}` grow exponentially from any
perturbation \u2014 which is exactly what the energy test was screaming: the energy was
growing as `exp(+\u03c9t)`, not oscillating.

The physical equation has a **minus sign**: `ddot{w} = -(D/\u03c1h) \u2207\u2074 w`. That's what
makes a plate ring rather than explode. One character.

I flipped the sign. All four tests passed. Energy drift over 2 ms: 1.4%.

## Why this is the interview story

Three things, in order of importance:

1. **The test caught it.** I didn't find the bug by staring at waveforms \u2014 a diverging
   solver can produce output that *looks* like a burst before it overflows. I found it
   because I'd written a conservation law first and the law refused to hold. That's the
   whole argument for numeric tests over visual inspection.

2. **The wrong trails were the instructive ones.** CFL, boundaries, spectral radius \u2014
   each was a reasonable hypothesis, each was checkable, each was wrong. The discipline
   of *falsifying the cheap hypothesis before the expensive one* is the part of the job
   that doesn't show up in a commit message.

3. **The fix was one character.** The hardest bugs are not the long ones; they're the
   ones where the surrounding code is correct enough that the obvious checks pass. A
   sign error in a PDE with an energy test is the canonical example.

## What I'd say if asked "what was the hardest bug you found"

> "I wrote an FDTD solver for a thin plate and an energy-conservation test for it
> before I had a working source. The test failed on the first run \u2014 the energy was
> growing exponentially instead of oscillating. I checked the CFL condition
> analytically and numerically, then the boundary handling, then the spectral radius
> of the discrete operator; all correct. The bug was a sign error in the equation of
> motion \u2014 I'd written `ddot{w} = +D \u2207\u2074 w` instead of `-D \u2207\u2074 w`. One character.
> The conservation test caught it; visual inspection of the waveforms would not have."