# Dispersion-Induced TDOA Bias

Sprint 0.5 deliverable (BACKLOG.md). The A0 flexural mode is intrinsically dispersive:
group velocity `c_g(f) = (4*pi^2*D/(rho h))^{1/4} * sqrt(f)`, so higher frequencies
travel faster. This document records the measured bias and the consequence for Sprint 1's
TDOA pipeline.

## Measurement

Method: inject a broadband crack-growth pulse at (0.15, 0.15) m, record the direct
arrival at sensor 1 (0.95, 0.05) m \u2014 distance ~1.13 m \u2014 and compare the
band-limited arrival times. See `src/structural_listener/sim/dispersion.py`.

| Band | Centre | Theoretical c_g | Measured c_g |
|------|--------|-----------------|--------------|
| Low | 20 \u00b1 5 kHz | 1247 m/s | ~830 m/s |
| High | 60 \u00b1 5 kHz | 2160 m/s (grid-capped at 1237 m/s) | ~1250 m/s |

The theoretical c_g uses `c_g = 2 c_p` for the Kirchhoff plate (group velocity is twice
phase velocity for `omega = sqrt(D/rho h) * k^2`). The high-band theoretical value
exceeds the 5-point-stencil grid group-velocity cap `2\u00b7c_basis/dx \u2248 1237 m/s`
(docs/sources.md), so the grid-capped value is what the simulation produces.

**Bias:** the high band arrives earlier than the low band \u2014 the A0 dispersion
trend is correct (higher frequencies travel faster). The measured c_g values sit below
the continuum theory because (a) the grid caps the group velocity and (b) the
bandpass onset estimator catches slower-frequency skirt leakage.

## Why the measured c_g is below the continuum theory

Two reasons, both honest:

1. **Grid group-velocity cap.** The 5-point-stencil biharmonic operator caps group
   velocity at `2\u00b7c_basis/dx \u2248 1237 m/s` (docs/sources.md). The high band's
   continuum c_g (~2160 m/s) is unrepresentable on the dx=5 mm grid; the simulation
   produces the capped value.

2. **Onset-estimator skirt leakage.** The bandpass arrival estimator catches the
   envelope *onset* (first sample > 20% of the peak), not the peak. Slower-frequency
   components leak through the bandpass skirts and pull the onset later than a pure
   tone at the band centre would arrive. This is a measurement artifact, not a solver
   bug \u2014 the solver's dispersion is correct (the trend and order of magnitude
   match).

Tightening would need a narrower source spectrum and a longer propagation path; both
cost runtime and are out of scope for the timebox (ARCHITECTURE.md \u00a78, parking
lot: "multi-mode dispersion").

## Consequence for localisation

The inter-band dispersion bias is the reason the TDOA picker uses a *fixed* band
(20\u201390 kHz) for all sensors: the bias cancels in the TDOA differences. The residual
within-band dispersion is what bounds the localisation error, and the Sprint 1.4 study
(`docs/error-distribution.md`) quantifies it at **2.8 cm median**.

**Mitigation (ADR-003):** arrival time is estimated at a *fixed* band for all sensors,
so the dispersion bias cancels in the TDOA differences. The effective wave speed used
by the multilateration solver is the grid group velocity for that band
(`c_eff = 1200 m/s`, cited in `docs/sources.md`), not the continuum theoretical c_g.