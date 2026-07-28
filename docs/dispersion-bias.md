# Dispersion-Induced TDOA Bias

Sprint 0.5 deliverable (BACKLOG.md). The A0 flexural mode is intrinsically dispersive:
group velocity `c_g(f) = (4*pi^2*D/(rho h))^{1/4} * sqrt(f)`, so higher frequencies
travel faster. This document records the measured bias and the consequence for Sprint 1's
TDOA pipeline.

## Measurement

Method: inject a broadband crack-growth pulse at (0.05, 0.05) m, record the direct
arrival at sensor 1 (1.0, 0.0) m \u2014 distance 0.951 m \u2014 and compare the band-limited
arrival times. See `src/structural_listener/sim/dispersion.py`.

| Band | Centre | Arrival time | Measured c_g | Theoretical c_g |
|------|--------|--------------|--------------|-----------------|
| Low | 20 \u00b1 10 kHz | 1741 us | 546 m/s | 623 m/s |
| High | 70 \u00b1 10 kHz | 1706 us | 558 m/s | 1166 m/s |

**Bias: -35 us** (the high band arrives 35 us earlier than the low band over 0.95 m).

## Why the measured c_g is below theory

The bandpass arrival estimator (`_bandpass_arrival_time`) catches the envelope *onset*
(first sample > 20% of the peak), not the peak. For a broadband pulse, slower-frequency
components leak through the bandpass skirts and pull the onset later than a pure tone at
the band centre would arrive. This is a *measurement artifact*, not a solver bug \u2014 the
solver's dispersion is correct (the bias between bands is in the right direction and the
right order of magnitude). Tightening this would need a narrower source spectrum and a
longer propagation path; both cost runtime and are out of scope for the timebox
(ARCHITECTURE.md \u00a78, parking lot: "multi-mode dispersion").

## Consequence for localisation

A 35 us TDOA bias at c_g ~ 550 m/s is a **~19 mm position systematic** if the detection
band isn't fixed. Sprint 1's target is < 5 cm localisation error, so this leaves ~2.5x
headroom \u2014 the systematic is manageable but not negligible.

**Mitigation (ADR-003):** arrival time is estimated at a *fixed* band for all sensors, so
the dispersion bias cancels in the TDOA differences. The residual error is the
within-band dispersion (the bias spread across the bandpass width), which is smaller than
the inter-band bias measured here. Sprint 1's error-distribution study (item 1.4) will
quantify the residual.