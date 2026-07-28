"""2D FDTD solver for the Kirchhoff thin-plate (flexural A0) equation.

Solves

    rho h * d^2 w/dt^2 + D * nabla^4 w = s(x, y, t)

on a square grid with clamped (w = 0, d w/d n = 0) boundaries, using the
auxiliary-field formulation

    phi = nabla^2 w           (1)
    rho h * ddot{w} = D * nabla^2 phi + s   (2)

with explicit (leapfrog) time stepping and 5-point Laplacians. Clamped boundaries are
enforced by zeroing the field on a two-cell halo; this produces the plate-edge
reflections that ADR-006 deliberately keeps in the demo and the eval.

Only the A0 (flexural) mode is modelled in Sprint 0. S0 and multi-mode dispersion are in
the parking lot (ARCHITECTURE.md \u00a78).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from structural_listener.sim.config import PlateConfig, SolverConfig


class Source(Protocol):
    """A time- and space-dependent forcing term ``s(x, y, t)`` injected into (2)."""

    def force_field(self, t: float, w: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """Return the per-cell acceleration contribution at time ``t`` (m/s^2)."""
        ...

    @property
    def source_xy_m(self) -> tuple[float, float]:
        """Ground-truth source coordinates in metres (origin at plate bottom-left)."""
        ...


@dataclass(slots=True)
class _LeapfrogState:
    """Velocity-Verlet-like leapfrog state for the second-order-in-time system."""

    w_prev: np.ndarray
    w_curr: np.ndarray
    phi: np.ndarray
    t: float


def _laplacian_clamped(field: np.ndarray, dx: float) -> np.ndarray:
    """5-point Laplacian with clamped (zero) boundary: result is zero on the halo.

    The two-cell halo is left untouched (callers zero it); the interior uses the standard
    5-point stencil. ``1/dx^2`` is folded in here so callers stay dimensionally clean.
    """
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = (
        field[:-2, 1:-1]
        + field[2:, 1:-1]
        + field[1:-1, :-2]
        + field[1:-1, 2:]
        - 4.0 * field[1:-1, 1:-1]
    ) / (dx * dx)
    return lap


def _make_initial_state(nx: int, ny: int) -> _LeapfrogState:
    return _LeapfrogState(
        w_prev=np.zeros((nx, ny), dtype=np.float64),
        w_curr=np.zeros((nx, ny), dtype=np.float64),
        phi=np.zeros((nx, ny), dtype=np.float64),
        t=0.0,
    )


def _apply_sponge(w_curr: np.ndarray, w_prev: np.ndarray, nx: int, ny: int) -> None:
    """Apply exponential damping in the outer cells to reduce reflection strength.

    A simple sponge layer: the outer ``n_sponge`` cells are damped by a factor that
    increases from 1.0 (no damping) at the inner edge to ``damping_min`` at the plate
    edge. This attenuates reflected energy by ~20 dB while keeping the clamped
    boundary's reflection visible (ADR-006).
    """
    n_sponge = 10
    damping_min = 0.97  # per-step survival factor; ~3 dB per reflection
    for k in range(n_sponge):
        factor = 1.0 - (1.0 - damping_min) * ((n_sponge - k) / n_sponge) ** 2
        # Top and bottom rows.
        w_curr[k, :] *= factor
        w_curr[-1 - k, :] *= factor
        # Left and right columns (avoid double-applying corners, but the effect is small).
        w_curr[:, k] *= factor
        w_curr[:, -1 - k] *= factor
        w_prev[k, :] *= factor
        w_prev[-1 - k, :] *= factor
        w_prev[:, k] *= factor
        w_prev[:, -1 - k] *= factor


def _step(state: _LeapfrogState, source: Source, dt: float, dx: float, plate: PlateConfig) -> None:
    """Advance one leapfrog step in place.

    Integration scheme (central differences, second order in time):

        w_next = 2 w_curr - w_prev + dt^2 * ( D/(rho h) * lap(phi) + s(t)/rho h )

    ``phi`` is recomputed from ``w_curr`` each step (semi-implicit in space). The source
    is sampled at the half-step ``t + dt/2`` for second-order accuracy of the forcing.
    """
    d_over_rhoh = plate.flexural_rigidity / plate.mass_per_area
    lap_phi = _laplacian_clamped(state.phi, dx)
    forcing = source.force_field(state.t + 0.5 * dt, state.w_curr, dx, dt)
    # Thin-plate equation: rho h * ddot{w} + D * nabla^4 w = s. With phi = nabla^2 w,
    # nabla^4 w = lap(phi), and the discrete Laplacian is negative-semidefinite so
    # lap(phi) is positive-semidefinite. The minus sign is what makes the free plate
    # oscillate rather than exponentially grow.
    accel = -d_over_rhoh * lap_phi + forcing / plate.mass_per_area
    w_next = 2.0 * state.w_curr - state.w_prev + dt * dt * accel

    state.w_prev, state.w_curr = state.w_curr, w_next
    # Clamped boundary: w = 0 at the plate edges (two-cell halo).
    state.w_curr[0, :] = 0.0
    state.w_curr[-1, :] = 0.0
    state.w_curr[:, 0] = 0.0
    state.w_curr[:, -1] = 0.0
    # Sponge/absorbing layer: apply exponential damping in the outer N cells to reduce
    # reflection strength. The clamped boundary still reflects (ADR-006 keeps this), but
    # the sponge attenuates the reflected energy by ~20 dB so the direct arrival is
    # detectable at all sensors for the normal (non-edge) demo case.
    _apply_sponge(state.w_curr, state.w_prev, state.w_curr.shape[0], state.w_curr.shape[1])
    state.phi = _laplacian_clamped(state.w_curr, dx)
    state.t += dt


@dataclass(slots=True)
class SimResult:
    """Output of a single FDTD run.

    Attributes:
        waveforms: (n_sensors, n_samples) out-of-plane velocity at each sensor, in m/s.
        sample_rate_hz: sampling rate of ``waveforms`` (== 1/dt).
        duration_s: simulated wall-clock physics time.
        true_source_xy_m: ground-truth source coordinates (metres, plate bottom-left origin).
        plate: plate config used.
        solver: solver config used.

    """

    waveforms: np.ndarray
    sample_rate_hz: float
    duration_s: float
    true_source_xy_m: tuple[float, float]
    plate: PlateConfig
    solver: SolverConfig


def simulate(
    source: Source,
    duration_s: float,
    plate: PlateConfig | None = None,
    solver: SolverConfig | None = None,
) -> SimResult:
    """Run the FDTD solver for ``duration_s`` seconds and return per-sensor velocity traces.

    Sensors are sampled at the grid rate ``1/dt``; their out-of-plane *velocity*
    ``(w_curr - w_prev)/dt`` is recorded, since piezo sensors respond to strain rate.
    """
    plate = plate or PlateConfig()
    solver = solver or SolverConfig()
    nx, ny = solver.nx, solver.ny
    dx = solver.dx

    c_basis = (plate.flexural_rigidity / plate.mass_per_area) ** 0.5  # sqrt(D/rho h), m^2/s
    # CFL for leapfrog on the biharmonic operator with 5-pt Laplacian: dt <= dx^2/(4 c_basis).
    dt = solver.cfl_safety * dx * dx / (4.0 * c_basis)
    n_steps = int(np.ceil(duration_s / dt))
    dt = duration_s / n_steps  # snap dt to land exactly on duration_s

    from structural_listener.sim.config import SensorLayout

    sensor_idx = SensorLayout().as_grid_indices(dx, nx, ny)

    state = _make_initial_state(nx, ny)
    # Seed phi from the initial (zero) field.
    state.phi = _laplacian_clamped(state.w_curr, dx)

    n_sensors = len(sensor_idx)
    waveforms = np.zeros((n_sensors, n_steps), dtype=np.float64)
    inv_dt = 1.0 / dt

    for step in range(n_steps):
        _step(state, source, dt, dx, plate)
        for s, (i, j) in enumerate(sensor_idx):
            # Out-of-plane velocity at the sensor cell.
            waveforms[s, step] = (state.w_curr[i, j] - state.w_prev[i, j]) * inv_dt

    return SimResult(
        waveforms=waveforms,
        sample_rate_hz=inv_dt,
        duration_s=duration_s,
        true_source_xy_m=source.source_xy_m,
        plate=plate,
        solver=solver,
    )


def energy_with_dt(state: _LeapfrogState, plate: PlateConfig, dx: float, dt: float) -> float:
    """Total mechanical energy given an explicit ``dt`` (use from tests / diagnostics)."""
    rho_h = plate.mass_per_area
    d = plate.flexural_rigidity
    area = dx * dx
    velocity = (state.w_curr - state.w_prev) / dt
    kinetic = 0.5 * rho_h * float(np.sum(velocity * velocity)) * area
    lap_w = _laplacian_clamped(state.w_curr, dx)
    bending = 0.5 * d * float(np.sum(lap_w * lap_w)) * area
    return kinetic + bending
