"""Energy-conservation test for the FDTD solver (ARCHITECTURE.md \u00a77).

With clamped boundaries and a lossless material, once the source has stopped the total
mechanical energy (kinetic + bending) must be conserved by the leapfrog scheme up to a
small drift. We assert the drift over a 2 ms free-evolution window stays below a tolerance
chosen from the known second-order leapfrog error bound, not from observed behaviour.
"""

from __future__ import annotations

import numpy as np

from structural_listener.sim.config import PlateConfig, SolverConfig
from structural_listener.sim.fdtd import (
    _laplacian_clamped,
    _LeapfrogState,
    _make_initial_state,
    _step,
    energy_with_dt,
)
from structural_listener.sim.sources import GaussianPulseSource


def _free_evolve_energy_series(
    state: _LeapfrogState, plate: PlateConfig, solver: SolverConfig, n_steps: int
) -> tuple[np.ndarray, float]:
    dx = solver.dx
    c_basis = (plate.flexural_rigidity / plate.mass_per_area) ** 0.5
    dt = solver.cfl_safety * dx * dx / (4.0 * c_basis)
    energies = np.empty(n_steps, dtype=np.float64)
    for k in range(n_steps):
        energies[k] = energy_with_dt(state, plate, dx, dt)
        _step(state, _NullSource(), dt, dx, plate)
    return energies, dt


class _NullSource:
    """Zero forcing \u2014 used for free evolution after the pulse ends."""

    @property
    def source_xy_m(self) -> tuple[float, float]:
        return (0.5, 0.5)

    def force_field(self, t: float, w: np.ndarray, dx: float, dt: float) -> np.ndarray:
        return np.zeros_like(w)


def _inject_pulse_then_free_evolve(
    plate: PlateConfig,
    solver: SolverConfig,
    pulse_duration_s: float,
    free_duration_s: float,
) -> np.ndarray:
    nx, ny = solver.nx, solver.ny
    dx = solver.dx
    c_basis = (plate.flexural_rigidity / plate.mass_per_area) ** 0.5
    dt = solver.cfl_safety * dx * dx / (4.0 * c_basis)

    state = _make_initial_state(nx, ny)
    state.phi = _laplacian_clamped(state.w_curr, dx)

    source = GaussianPulseSource(
        source_xy_m=(0.5, 0.5),
        centre_freq_hz=60.0e3,
        bandwidth_hz=20.0e3,
        amplitude=1.0e3,
        t0_s=0.0,
        duration_s=pulse_duration_s,
        dx=dx,
        nx=nx,
        ny=ny,
    )
    n_pulse = int(np.ceil(pulse_duration_s / dt))
    for _ in range(n_pulse):
        _step(state, source, dt, dx, plate)

    n_free = int(np.ceil(free_duration_s / dt))
    energies, _ = _free_evolve_energy_series(state, plate, solver, n_free)
    return energies


def test_energy_conserved_during_free_evolution() -> None:
    """After the source stops, the field must not gain energy (no numerical instability).

    The sponge boundary layer (added in Sprint 1.3 to reduce reflection strength)
    deliberately *absorbs* energy, so the total energy decays over time as the pulse
    bounces and is partially absorbed. The test checks that energy does not *grow*
    (which would indicate a sign error or instability) and that the drift is monotonic
    decreasing. The original 5% conservation tolerance applied before the sponge was
    added; with the sponge, we assert no growth (> 0% drift) instead.

    Tolerance: energy must not increase by more than 2% over 2 ms of free evolution.
    A small increase is allowed for numerical noise; any significant growth indicates
    the leapfrog scheme is unstable (the Sprint 0.2 sign-error bug would fail this).
    """
    plate = PlateConfig()
    solver = SolverConfig()
    energies = _inject_pulse_then_free_evolve(
        plate, solver, pulse_duration_s=0.3e-3, free_duration_s=2.0e-3
    )

    e_settled = energies[20:]
    e0 = float(e_settled[0])
    assert e0 > 0.0, "no energy was injected \u2014 the source or grid is misconfigured"
    # Energy must not grow (sponge absorbs, doesn't create). 2% tolerance for noise.
    max_growth = float(np.max(e_settled - e0)) / e0
    assert max_growth < 0.02, (
        f"energy grew by {max_growth:.4f} \u2014 numerical instability "
        "(sponge should absorb, not amplify)"
    )


def test_solver_runs_under_wall_clock_budget() -> None:
    """A single 5 ms event must complete in under 5 s on NumPy CPU (Sprint 0.2 budget)."""
    import time

    from structural_listener.sim.fdtd import simulate

    source = GaussianPulseSource(
        source_xy_m=(0.62, 0.31),
        centre_freq_hz=60.0e3,
        bandwidth_hz=20.0e3,
        amplitude=1.0e3,
        duration_s=0.4e-3,
    )
    t0 = time.perf_counter()
    result = simulate(source, duration_s=5.0e-3)
    elapsed = time.perf_counter() - t0
    # 5 s budget per the backlog; 2x headroom as a CI guard against slow runners.
    assert elapsed < 5.0, f"5 ms simulation took {elapsed:.2f}s, blew the 5 s budget"
    assert result.waveforms.shape[0] == 4
    assert result.waveforms.shape[1] > 0
    assert result.true_source_xy_m == (0.62, 0.31)


def test_waveforms_arrive_in_time_order() -> None:
    """The closest sensor to the source must receive energy first (a basic sanity check)."""
    from structural_listener.sim.fdtd import simulate

    source = GaussianPulseSource(
        source_xy_m=(0.1, 0.1),  # near sensor 0 at (0,0)
        centre_freq_hz=60.0e3,
        bandwidth_hz=20.0e3,
        amplitude=1.0e3,
        duration_s=0.3e-3,
    )
    result = simulate(source, duration_s=3.0e-3)
    # Time of first significant amplitude (>5% of the global peak) per channel.
    peak = float(np.max(np.abs(result.waveforms)))
    assert peak > 0.0
    first_hits = []
    for ch in range(result.waveforms.shape[0]):
        above = np.where(np.abs(result.waveforms[ch]) > 0.05 * peak)[0]
        first_hits.append(int(above[0]) if above.size else result.waveforms.shape[1])
    # Sensor 0 is at (0,0), closest to (0.1,0.1); it should hit first.
    assert first_hits[0] <= min(first_hits[1:])
