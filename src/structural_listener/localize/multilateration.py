"""Closed-form multilateration by Gauss-Newton least squares.

Given arrival times ``t_i`` at sensors ``s_i`` and wave speed ``c``, solve for the
source position ``p`` that minimises the TDOA residual:

    minimise  sum_{i,j}  [ (|p - s_i| - |p - s_j|)/c  -  (t_i - t_j) ]^2

This is the formulation from ARCHITECTURE.md \u00a74, derivable on a whiteboard
(ADR-003). Gauss-Newton iteration from a centroid initial guess converges in
~5-10 iterations for a 1 m plate. The residual at convergence is reported as a
confidence score: a high residual means the arrival times are geometrically
inconsistent (usually a reflection was mistaken for the direct arrival \u2014 ADR-006),
and the system reports low confidence rather than a confidently wrong coordinate.

No machine learning. No training data. No black box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MultilaterationConfig:
    """Parameters for the Gauss-Newton solver."""

    # Maximum Gauss-Newton iterations. Convergence on a 1 m plate is ~5-10.
    max_iterations: int = 50
    # Convergence threshold: stop when the position update is below this (metres).
    tolerance_m: float = 1.0e-4
    # Step damping factor. 1.0 = pure Gauss-Newton; < 1.0 adds gradient-descent
    # robustness for poor initial guesses. We use 0.5 for stability.
    step_damping: float = 0.5


@dataclass(frozen=True, slots=True)
class LocalisationResult:
    """The solver's answer.

    Attributes:
        position_m: estimated (x, y) source coordinates.
        residual: the RMS TDOA residual at convergence, in seconds. This is the
            confidence score \u2014 low is good.
        confidence: 1 / (1 + residual / confidence_scale), in [0, 1]. A high residual
            gives low confidence (ADR-006: report uncertainty, not a wrong coordinate).
        converged: whether Gauss-Newton reached the tolerance within max_iterations.
        iterations: number of iterations performed.
        used_channels: the sensor indices used in the solve.

    """

    position_m: tuple[float, float]
    residual: float
    confidence: float
    converged: bool
    iterations: int
    used_channels: tuple[int, ...]


def _residual_vector(
    position: np.ndarray,
    sensor_positions: np.ndarray,
    arrival_times: np.ndarray,
    wave_speed: float,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Compute the TDOA residual vector: r_k = (|p - s_i| - |p - s_j|)/c - (t_i - t_j)."""
    distances = np.sqrt(np.sum((sensor_positions - position) ** 2, axis=1))
    r = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        predicted_tdoa = (distances[i] - distances[j]) / wave_speed
        measured_tdoa = arrival_times[i] - arrival_times[j]
        r[k] = predicted_tdoa - measured_tdoa
    return r


def _jacobian(
    position: np.ndarray,
    sensor_positions: np.ndarray,
    wave_speed: float,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Jacobian of the residual w.r.t. position (x, y).

    dr_k/dp = (1/c) * [ (p - s_i)/|p - s_i| - (p - s_j)/|p - s_j| ]
    """
    distances = np.sqrt(np.sum((sensor_positions - position) ** 2, axis=1))
    # Unit vectors from each sensor to the position.
    unit = np.zeros_like(sensor_positions)
    for i in range(sensor_positions.shape[0]):
        if distances[i] > 1e-12:
            unit[i] = (position - sensor_positions[i]) / distances[i]
    jac = np.zeros((len(pairs), 2))
    for k, (i, j) in enumerate(pairs):
        jac[k] = (unit[i] - unit[j]) / wave_speed
    return jac


def solve(
    sensor_positions: np.ndarray,
    arrival_times: np.ndarray,
    wave_speed: float,
    config: MultilaterationConfig | None = None,
    active_channels: tuple[int, ...] | None = None,
) -> LocalisationResult:
    """Solve for the source position from arrival times at known sensors.

    Args:
        sensor_positions: (n_sensors, 2) array of sensor (x, y) coordinates in metres.
        arrival_times: (n_sensors,) array of arrival times in seconds. Channels with
            no detected hit should be set to NaN; they are excluded.
        wave_speed: the effective wave speed in m/s. Use a calibrated value, not the
            theoretical c_g \u2014 the dispersion bias is absorbed here (ADR-003).
        config: solver parameters.
        active_channels: if provided, only these channel indices are used. If None,
            all channels with non-NaN arrival times are used.

    Returns:
        A ``LocalisationResult`` with the estimated position, residual, and confidence.

    Raises:
        ValueError: if fewer than 3 channels have valid arrival times (multilateration
            needs at least 3 sensors for a 2D position).

    """
    cfg = config or MultilaterationConfig()

    if active_channels is None:
        active_channels = tuple(
            i for i in range(len(arrival_times)) if not np.isnan(arrival_times[i])
        )
    if len(active_channels) < 3:
        raise ValueError(
            f"multilateration needs >= 3 sensors with valid arrival times, "
            f"got {len(active_channels)}"
        )

    sensors = np.asarray(sensor_positions)[list(active_channels)]
    times = np.asarray(arrival_times)[list(active_channels)]

    # Use the earliest arrival as the reference; TDOA pairs are (i, 0) for i > 0.
    ref_idx = int(np.argmin(times))
    pairs = [(i, ref_idx) for i in range(len(active_channels)) if i != ref_idx]

    # Initial guess: the centroid of the active sensors.
    position = np.mean(sensors, axis=0).astype(np.float64)

    converged = False
    iterations = 0
    for it in range(cfg.max_iterations):
        iterations = it + 1
        r = _residual_vector(position, sensors, times, wave_speed, pairs)
        jac = _jacobian(position, sensors, wave_speed, pairs)
        # Gauss-Newton update: delta = -(J^T J)^{-1} J^T r, with damping.
        jtj = jac.T @ jac
        jtr = jac.T @ r
        try:
            delta = cfg.step_damping * np.linalg.solve(jtj, -jtr)
        except np.linalg.LinAlgError:
            # Singular matrix \u2014 add a small regularisation and retry.
            delta = cfg.step_damping * np.linalg.solve(jtj + 1e-10 * np.eye(2), -jtr)
        position = position + delta
        if float(np.linalg.norm(delta)) < cfg.tolerance_m:
            converged = True
            break

    # Final residual (RMS).
    r_final = _residual_vector(position, sensors, times, wave_speed, pairs)
    residual = float(np.sqrt(np.mean(r_final**2)))

    # Confidence: map residual to [0, 1]. A residual of 50 us -> ~0.5 confidence.
    # 50 us at c=1000 m/s is ~5 cm \u2014 the Sprint 1 accuracy target. So a residual
    # at the target gives 50% confidence; lower residuals give higher confidence.
    confidence_scale = 50.0e-6
    confidence = 1.0 / (1.0 + residual / confidence_scale)

    return LocalisationResult(
        position_m=(float(position[0]), float(position[1])),
        residual=residual,
        confidence=confidence,
        converged=converged,
        iterations=iterations,
        used_channels=active_channels,
    )
