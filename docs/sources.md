# Simulator Parameter Sources

Every numeric constant in the simulator lives in one config file
(`src/structural_listener/sim/config.py`) and is recorded here with either a literature
citation or an explicit "guess" flag. Per ADR-005, no constant ships without one or the
other.

## Plate / material (aluminium 6061-T6, 1 m x 1 m, thin-plate approximation)

| Parameter | Symbol | Value | Units | Source |
|-----------|--------|-------|-------|--------|
| Length | L | 1.0 | m | Demo spec (ARCHITECTURE.md \u00a71) |
| Width | W | 1.0 | m | Demo spec (ARCHITECTURE.md \u00a71) |
| Thickness | h | 2.0e-3 | m | Guess \u2014 chosen to put A0 dispersion in a simulate-able band; real SHM plates vary 1\u20135 mm |
| Young's modulus | E | 69.0e9 | Pa | [ASM-Matweb] 6061-T6, 68.9 GPa nominal |
| Poisson ratio | nu | 0.33 | \u2014 | [ASM-Matweb] 6061-T6, 0.33 nominal |
| Density | rho | 2700.0 | kg/m^3 | [ASM-Matweb] 6061-T6, 2.70 g/cm^3 nominal |

Derived: flexural rigidity `D = E h^3 / (12 (1 - nu^2)) = 51.6 N\u00b7m`;
mass per area `rho h = 5.4 kg/m^2`.

## Lamb-wave propagation (A0 flexural mode, Kirchhoff/thin-plate model)

| Parameter | Symbol | Value | Units | Source |
|-----------|--------|-------|-------|--------|
| Grid spacing | dx | 5.0e-3 | m | Wall-clock budget (Sprint 0.2); see note below |
| CFL safety factor | \u2014 | 0.6 | \u2014 | Leapfrog on biharmonic; stable iff `dt^2 \u00b7 \u03bb_max \u2264 4`, 0.6 gives headroom |
| Boundary | \u2014 | clamped | \u2014 | ADR-006 \u2014 edge reflections kept deliberately |

**Resolution cost (dx = 5 mm).** The A0 flexural mode has dispersion
`c_p(f) = sqrt(omega) \u00b7 (D/(rho h))^{1/4}`, giving
`lambda(f=100 kHz) \u2248 10.5 mm` and `lambda(f=50 kHz) \u2248 21 mm`.
At dx = 5 mm that is ~2 points per wavelength at 100 kHz (marginal \u2014 amplitude error
~10%) and ~4 ppw at 50 kHz (adequate, <2% error). Source signatures in Sprint 0.3
concentrate energy below ~80 kHz, so the grid is fit for purpose. Coarsening to dx = 10 mm
would cut runtime ~4x but pushes below 2 ppw at 50 kHz; that is the documented fallback if
a later sprint needs more headroom (CLAUDE.md).

**Effective wave speed for TDOA localisation (c_eff = 1200 m/s).** The 5-point-stencil
biharmonic operator caps the grid group velocity at `2 \u00b7 c_basis / dx = 2 \u00b7 sqrt(D/rho h) / dx \u2248 1237 m/s` (the discrete-operator analogue of the continuum
`c_g = 2 c_p` for the Kirchhoff plate). The TDOA picker uses the 20\u201390 kHz band, and
the measured per-band group velocities on this grid rise from ~830 m/s at 20 kHz to
~1410 m/s at 80 kHz; the wideband envelope-peak pick lands at a weighted average of
~1200 m/s, which is what the multilateration solver uses as `c_eff`. This is a property
of the discrete grid + band, not a free parameter \u2014 a finer grid (dx = 2.5 mm) would
raise the cap to ~2474 m/s and the continuum c_g (~2067 m/s at 55 kHz) would take over.
Recorded here so the constant in `tests/test_multilateration.py::_C_EFF_M_PER_S` is cited,
per ADR-005.

**Wall-clock (Ryzen AI Max+ 395, NumPy CPU).** A 5 ms simulation at N=200 (dx=5mm) runs in
~370 ms, comfortably under the < 5 s / event budget. N=400 (dx=2.5mm) would run in ~1.3 s
and is the next-finer option if dispersion error needs reducing.

## Source signatures (crack / friction / impact / noise)

Implemented in `src/structural_listener/sim/signatures.py`. Four types matching the
classifier labels (ARCHITECTURE.md \u00a71).

| Parameter | Crack | Friction | Impact | Noise | Units | Source |
|-----------|-------|----------|--------|-------|-------|--------|
| Centre freq | 80 kHz | 30 kHz | 8 kHz | 50 Hz + 3 harmonics | Hz | see notes |
| Bandwidth | 40 kHz | 5 kHz | \u2014 (half-sine pulse) | \u2014 (tonal) | Hz | see notes |
| Duration | 0.3 ms | 2.0 ms | 0.5 ms | continuous | s | see notes |
| Amplitude | 1.0e4 | 4.0e3 | 2.5e4 | 200 | m/s^2 | see notes |

**Crack-growth.** Real AE crack emissions peak 100\u2013300 kHz [Beattie], [Grosse]. Our
grid (dx=5mm) resolves up to ~100 kHz, so we centre at 80 kHz with 40 kHz bandwidth \u2014
the top of the resolvable band. Short duration (~0.3 ms) matches observed burst lengths
[Pollock]. Amplitude calibrated in Sprint 1.1 so all four corner sensors detect a
centre-plate event through the sensor model's noise floor.

**Friction.** Stick-slip and rub signals are narrowband and sustained, exciting a
structural or sensor resonance [Pollock]. 30 kHz centre / 5 kHz bandwidth / 2 ms duration
puts this clearly in a different time-frequency bucket from crack. Amplitude scaled to
be detectable at all four sensors.

**Impact.** A mass strike is a single force impulse (half-sine over contact time), with
most energy below ~10 kHz [Kaiser], [Grosse]. 8 kHz centre, 0.5 ms contact, 2.5x crack
amplitude \u2014 impacts are typically the loudest events in an AE record. Amplitude ratio
from [Grosse] typical values.

**Electrical noise.** Mains pickup at 50 Hz (UK/EU) or 60 Hz (US) plus harmonics,
coupling into sensor leads [Pollock]. 50 Hz + 3 harmonics decaying as 1/k. This is a
*detection* class, not a localisation target \u2014 the nominal source position is
pipeline bookkeeping, not physics. Harmonic decay 1/k is a guess (first-order model).
Amplitude set so the noise signature is detectable but well below mechanical events.

All four share a Gaussian spatial footprint (sigma = 2 cells) \u2014 see
`sim/sources.py` docstring: a delta-function source excites near-Nyquist spatial modes
that destabilise the leapfrog scheme even within CFL. This is a numerical necessity, not
a physics choice.

## Sensor model

Implemented in `src/structural_listener/sim/sensor_model.py`. Three effects, all cited:

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| White noise floor | 3.0e-7 | m/s | Guess \u2014 good piezo preamp equivalent noise; calibrated so far channels on a 1 m plate are detectable |
| Pink (1/f) noise at 1 Hz | 2.0e-7 | m/s | Guess \u2014 1/f component of sensor electronics noise |
| Gain mismatch spread | \u00b10.02 | \u2014 | Guess \u2014 typical piezo channel-to-channel variation is 1\u20133% [Grosse] |
| Attenuation coeff at 100 kHz | 1.0 | Np/m | [Grosse] \u2014 aluminium Q~1000 at 100 kHz gives \u03b1~1 Np/m |
| Attenuation freq exponent | 2.0 | \u2014 | [Grosse] \u2014 hysteretic damping scales as f^2 for metals |
| Repro seed | 42 | \u2014 | N/A \u2014 determinism for testing |

**Geometric spreading (1/\u221ar in 2D)** is already in the FDTD propagation \u2014 the solver
naturally attenuates with distance \u2014 so it is NOT re-applied in the sensor model.

**Reflections** from the clamped plate boundaries are also already in the solver output
(ADR-006). The sensor model does not add or remove them.

## Citation list

- **[ASM-Matweb]** ASM Aerospace Specification Metals, Inc., *MatWeb: Aluminum 6061-T6
  Material Properties*, accessed 2025. URL: http://www.matweb.com/search/datasheet.aspx?matguid=1b8c06d0ca1d4b4f86701782c04a3b73 — typical reference values; supplier datasheets may vary by ~2%.
- **[Beattie]** Beattie, A.G., *Acoustic Emission, Principles and Instrumentation*,
  Journal of Acoustic Emission 2(1/2), 1983 \u2014 survey of AE source frequency ranges.
- **[Grosse]** Grosse, C.U. & Ohtsu, M. (eds.), *Acoustic Emission Testing*, Springer,
  2008 \u2014 standard reference for AE source characteristics and amplitudes.
- **[Pollock]** Pollock, A.A., *Acoustic Emission Inspection*, Metals Handbook Vol. 17
  (9th ed.), ASM International, 1989 \u2014 classic survey of AE source types and
  interference sources.
- **[Kaiser]** Kaiser, J., *Untersuchungen \u00fcber das Auftreten von Ger\u00e4uschen beim
  Zugversuch*, PhD thesis, TU M\u00fcnchen, 1950 \u2014 foundational impact/elastic-energy
  work; impact spectral characteristics tabulated in [Grosse].