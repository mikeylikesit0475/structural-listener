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

**Wall-clock (Ryzen AI Max+ 395, NumPy CPU).** A 5 ms simulation at N=200 (dx=5mm) runs in
~370 ms, comfortably under the < 5 s / event budget. N=400 (dx=2.5mm) would run in ~1.3 s
and is the next-finer option if dispersion error needs reducing.

## Source signatures (crack / friction / impact / noise)

_Pending Sprint 0.3. The Sprint 0.2 `GaussianPulseSource` is a stand-in with:_

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| Centre frequency | 60 kHz | Hz | Guess \u2014 inside the A0 band the grid resolves; real crack AE peaks 100\u2013300 kHz but our grid caps at ~100 kHz |
| Bandwidth | 20 kHz | Hz | Guess \u2014 gives a ~0.3 ms burst, short enough for clean TDOA |
| Amplitude | 1.0e3 | m/s^2 | Guess \u2014 arbitrary scale; Sprint 0.4 calibrates to a target sensor voltage |
| Spatial footprint | sigma = 2 cells | \u2014 | Numerical necessity \u2014 a delta source excites near-Nyquist modes that destabilise leapfrog even within CFL |

## Sensor model

_Pending Sprint 0.4._

## Citation list

- **[ASM-Matweb]** ASM Aerospace Specification Metals, Inc., *MatWeb: Aluminum 6061-T6
  Material Properties*, accessed 2025. URL: http://www.matweb.com/search/datasheet.aspx?matguid=1b8c06d0ca1d4b4f86701782c04a3b73 — typical reference values; supplier datasheets may vary by ~2%.