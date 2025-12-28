"""
Metrics Module for Inverted Pendulum Control System

This module provides functions to calculate performance metrics for controller
comparison. Metrics include settling time, overshoot, steady-state error,
control effort, and robustness measures.

All metrics are designed to be consistent and comparable across different
controller types and operating conditions.
"""

import numpy as np
import logging
from typing import Dict, Optional
from analysis import SimulationResult


logger = logging.getLogger(__name__)


def calculate_settling_time(
    time: np.ndarray,
    signal: np.ndarray,
    final_value: float = 0.0,
    tolerance: float = 0.02,
    min_settling_duration: float = 0.5
) -> float:
    """Calculate the settling time of a signal.

    Settling time is when the signal enters and stays within a tolerance
    band around the final value.

    Args:
        time: Time array (N,)
        signal: Signal array (N,)
        final_value: Target final value (default: 0.0)
        tolerance: Tolerance as fraction of final value (default: 2%)
        min_settling_duration: Minimum time signal must stay within tolerance (s)

    Returns:
        Settling time in seconds, or inf if never settles
    """
    # Calculate tolerance band
    # For final_value near zero, use absolute tolerance
    if abs(final_value) < 1e-6:
        tol_band = max(0.02, tolerance)  # Use absolute tolerance
    else:
        tol_band = abs(final_value * tolerance)

    # Find indices where signal is within tolerance
    within_tolerance = np.abs(signal - final_value) <= tol_band

    # Find the first time where signal stays within tolerance
    dt = time[1] - time[0] if len(time) > 1 else 0.01
    min_steps = int(min_settling_duration / dt)

    for i in range(len(time) - min_steps):
        # Check if signal stays within tolerance for min_settling_duration
        if np.all(within_tolerance[i:i + min_steps]):
            return time[i]

    # If never settles, return infinity
    return np.inf


def calculate_overshoot(
    signal: np.ndarray,
    final_value: float = 0.0
) -> float:
    """Calculate the maximum overshoot of a signal.

    Args:
        signal: Signal array (N,)
        final_value: Target final value (default: 0.0)

    Returns:
        Maximum overshoot as percentage
    """
    # Maximum deviation from final value
    max_deviation = np.max(np.abs(signal - final_value))

    # Overshoot as percentage
    if abs(final_value) < 1e-6:
        # For zero final value, return absolute overshoot
        return max_deviation * 100
    else:
        return (max_deviation / abs(final_value)) * 100


def calculate_steady_state_error(
    signal: np.ndarray,
    final_value: float = 0.0,
    last_percentage: float = 0.1
) -> float:
    """Calculate the steady-state error.

    Uses the mean of the last portion of the signal to determine
    steady-state value.

    Args:
        signal: Signal array (N,)
        final_value: Target final value (default: 0.0)
        last_percentage: Fraction of signal to use for steady-state (default: 10%)

    Returns:
        Steady-state error (absolute)
    """
    # Use last portion of signal
    n_last = max(1, int(len(signal) * last_percentage))
    steady_state_value = np.mean(signal[-n_last:])

    return abs(steady_state_value - final_value)


def calculate_control_effort(
    time: np.ndarray,
    force: np.ndarray
) -> Dict[str, float]:
    """Calculate control effort metrics.

    Args:
        time: Time array (N,)
        force: Force array (N,)

    Returns:
        Dictionary containing:
            - 'total_absolute': ∫|u(t)|dt (total absolute control effort)
            - 'total_squared': ∫u²(t)dt (total energy)
            - 'peak': max|u(t)| (peak force)
            - 'mean_absolute': mean(|u(t)|) (average effort)
    """
    dt = time[1] - time[0] if len(time) > 1 else 0.01

    total_absolute = np.trapz(np.abs(force), dx=dt)
    total_squared = np.trapz(force**2, dx=dt)
    peak = np.max(np.abs(force))
    mean_absolute = np.mean(np.abs(force))

    return {
        'total_absolute': total_absolute,
        'total_squared': total_squared,
        'peak': peak,
        'mean_absolute': mean_absolute
    }


def calculate_rms_error(
    signal: np.ndarray,
    reference: float = 0.0
) -> float:
    """Calculate root-mean-square error.

    Args:
        signal: Signal array (N,)
        reference: Reference value (default: 0.0)

    Returns:
        RMS error
    """
    error = signal - reference
    return np.sqrt(np.mean(error**2))


def calculate_all_metrics(result: SimulationResult) -> Dict[str, float]:
    """Calculate comprehensive performance metrics for a simulation result.

    Args:
        result: SimulationResult object

    Returns:
        Dictionary of performance metrics
    """
    logger.info(f"Calculating metrics for {result.get_label()}")

    metrics = {}

    # Extract signals
    time = result.time
    states = result.states
    forces = result.forces

    # State components
    x = states[:, 0]          # Cart position
    x_dot = states[:, 1]      # Cart velocity
    theta = states[:, 2]      # Pendulum angle
    theta_dot = states[:, 3]  # Pendulum angular velocity

    # Angle metrics (primary control objective)
    metrics['angle_settling_time'] = calculate_settling_time(
        time, theta, final_value=0.0, tolerance=0.02
    )
    metrics['angle_overshoot'] = calculate_overshoot(theta, final_value=0.0)
    metrics['angle_steady_state_error'] = calculate_steady_state_error(theta, final_value=0.0)
    metrics['angle_rms'] = calculate_rms_error(theta, reference=0.0)

    # Position metrics
    metrics['position_settling_time'] = calculate_settling_time(
        time, x, final_value=0.0, tolerance=0.05
    )
    metrics['position_final'] = np.mean(x[-100:])  # Final position
    metrics['position_rms'] = calculate_rms_error(x, reference=0.0)

    # Control effort metrics
    effort_metrics = calculate_control_effort(time, forces)
    metrics['control_effort_total'] = effort_metrics['total_absolute']
    metrics['control_effort_squared'] = effort_metrics['total_squared']
    metrics['control_effort_peak'] = effort_metrics['peak']
    metrics['control_effort_mean'] = effort_metrics['mean_absolute']

    # Additional robustness metrics
    metrics['max_angle'] = np.max(np.abs(theta))
    metrics['max_position'] = np.max(np.abs(x))
    metrics['max_angular_velocity'] = np.max(np.abs(theta_dot))

    # Success metric (did it stabilize?)
    angle_settled = metrics['angle_settling_time'] < np.inf
    position_bounded = metrics['max_position'] < 5.0  # Cart stayed within 5m
    metrics['success'] = 1.0 if (angle_settled and position_bounded) else 0.0

    # Store in result object
    result.metrics = metrics

    logger.info(f"  Settling time: {metrics['angle_settling_time']:.3f} s")
    logger.info(f"  Total control effort: {metrics['control_effort_total']:.2f}")
    logger.info(f"  Success: {metrics['success']}")

    return metrics


def compare_metrics(results: list) -> None:
    """Print a comparison table of metrics across multiple results.

    Args:
        results: List of SimulationResult objects with metrics calculated
    """
    if not results:
        logger.warning("No results to compare")
        return

    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON")
    print("=" * 100)

    # Table header
    header = f"{'Configuration':<30} {'Settling (s)':<15} {'Control Effort':<15} {'Peak Force':<12} {'Success':<10}"
    print(header)
    print("-" * 100)

    # Table rows
    for result in results:
        if not result.metrics:
            logger.warning(f"No metrics calculated for {result.get_label()}")
            continue

        label = result.get_label()
        settling = result.metrics['angle_settling_time']
        effort = result.metrics['control_effort_total']
        peak = result.metrics['control_effort_peak']
        success = "Yes" if result.metrics['success'] == 1.0 else "No"

        settling_str = f"{settling:.3f}" if settling < np.inf else "DNF"

        row = f"{label:<30} {settling_str:<15} {effort:<15.2f} {peak:<12.2f} {success:<10}"
        print(row)

    print("=" * 100)

    # Summary statistics
    print("\nKEY INSIGHTS:")

    # Find best performers
    successful_results = [r for r in results if r.metrics.get('success', 0) == 1.0]

    if successful_results:
        # Best settling time
        best_settling = min(successful_results, key=lambda r: r.metrics['angle_settling_time'])
        print(f"  Fastest settling: {best_settling.get_label()} "
              f"({best_settling.metrics['angle_settling_time']:.3f} s)")

        # Most efficient
        best_effort = min(successful_results, key=lambda r: r.metrics['control_effort_total'])
        print(f"  Most efficient: {best_effort.get_label()} "
              f"(effort = {best_effort.metrics['control_effort_total']:.2f})")

        # Comparison: LQR vs PID
        lqr_results = [r for r in successful_results if 'LQR' in r.controller_type]
        pid_results = [r for r in successful_results if 'PID' in r.controller_type]

        if lqr_results and pid_results:
            lqr_avg_effort = np.mean([r.metrics['control_effort_total'] for r in lqr_results])
            pid_avg_effort = np.mean([r.metrics['control_effort_total'] for r in pid_results])
            effort_reduction = ((pid_avg_effort - lqr_avg_effort) / pid_avg_effort) * 100

            print(f"\n  LQR vs PID:")
            print(f"    LQR average effort: {lqr_avg_effort:.2f}")
            print(f"    PID average effort: {pid_avg_effort:.2f}")
            print(f"    LQR effort reduction: {effort_reduction:.1f}%")
    else:
        print("  No successful stabilizations!")

    print()
