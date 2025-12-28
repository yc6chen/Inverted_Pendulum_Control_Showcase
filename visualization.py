"""
Visualization Module for Inverted Pendulum Control System

This module provides publication-quality plotting functions for comparing
controller performance. It implements Phase 5's visualization requirements
with a consistent style and clear data presentation.

Functions:
- plot_angle_comparison: Compare angle trajectories across scenarios
- plot_force_comparison: Compare control forces
- plot_position_comparison: Compare cart positions
- plot_tradeoff_scatter: Settling time vs control effort tradeoff
- generate_comprehensive_report: Create complete analysis figure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from typing import List, Dict, Optional, Tuple
import logging

from analysis import SimulationResult


logger = logging.getLogger(__name__)


# Set consistent plotting style
try:
    mplstyle.use('seaborn-v0_8-whitegrid')
except:
    # Fallback if seaborn style not available
    mplstyle.use('default')
    plt.style.use('bmh')


# Color palette for different scenarios
COLORS = {
    'PID-Ideal': '#2E86AB',        # Blue
    'PID-Noisy': '#A23B72',        # Purple
    'PID-Noisy+KF': '#F18F01',     # Orange
    'LQR-Ideal': '#06A77D',        # Green
    'LQR-Noisy': '#D00000',        # Red
    'LQR-Noisy+KF': '#6A0572',     # Deep purple
    'CASCADE_PID-Ideal': '#0496FF',
    'CASCADE_PID-Noisy+KF': '#FB6107',
    'GAIN_SCHEDULED_PID-Ideal': '#9D4EDD',
    'GAIN_SCHEDULED_PID-Noisy+KF': '#3A86FF'
}

# Line styles
LINESTYLES = {
    'Ideal': '-',
    'Noisy': '--',
    'Noisy+KF': '-.'
}


def _get_plot_style(label: str) -> Tuple[str, str, float]:
    """Get consistent color and linestyle for a result label.

    Args:
        label: Result label (e.g., 'PID-Ideal')

    Returns:
        Tuple of (color, linestyle, linewidth)
    """
    # Extract noise/filter info
    if 'Noisy+KF' in label:
        style_key = 'Noisy+KF'
    elif 'Noisy' in label:
        style_key = 'Noisy'
    else:
        style_key = 'Ideal'

    color = COLORS.get(label, '#333333')  # Default to dark gray
    linestyle = LINESTYLES.get(style_key, '-')
    linewidth = 2.0 if 'Ideal' in label else 1.5

    return color, linestyle, linewidth


def plot_angle_comparison(
    results: List[SimulationResult],
    ax: Optional[plt.Axes] = None,
    title: str = "Pendulum Angle Comparison"
) -> plt.Axes:
    """Plot pendulum angle trajectories for multiple scenarios.

    Args:
        results: List of SimulationResult objects
        ax: Optional matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        label = result.get_label()
        color, linestyle, linewidth = _get_plot_style(label)

        # Convert angle to degrees for readability
        angle_deg = np.rad2deg(result.states[:, 2])

        ax.plot(
            result.time,
            angle_deg,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8
        )

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Pendulum Angle (degrees)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)

    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    return ax


def plot_force_comparison(
    results: List[SimulationResult],
    ax: Optional[plt.Axes] = None,
    title: str = "Control Force Comparison",
    show_saturation: bool = True
) -> plt.Axes:
    """Plot control force trajectories for multiple scenarios.

    Args:
        results: List of SimulationResult objects
        ax: Optional matplotlib axes (creates new if None)
        title: Plot title
        show_saturation: Whether to show saturation limits

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot saturation limits if requested
    if show_saturation and results:
        saturation_limits = results[0].config.control.saturation_limits
        if saturation_limits is not None:
            ax.axhline(
                y=saturation_limits[1],
                color='red',
                linestyle=':',
                linewidth=1.5,
                label='Saturation Limits',
                alpha=0.6
            )
            ax.axhline(
                y=saturation_limits[0],
                color='red',
                linestyle=':',
                linewidth=1.5,
                alpha=0.6
            )

    for result in results:
        label = result.get_label()
        color, linestyle, linewidth = _get_plot_style(label)

        ax.plot(
            result.time,
            result.forces,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8
        )

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Control Force (N)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)

    return ax


def plot_position_comparison(
    results: List[SimulationResult],
    ax: Optional[plt.Axes] = None,
    title: str = "Cart Position Comparison"
) -> plt.Axes:
    """Plot cart position trajectories for multiple scenarios.

    Args:
        results: List of SimulationResult objects
        ax: Optional matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        label = result.get_label()
        color, linestyle, linewidth = _get_plot_style(label)

        position = result.states[:, 0]

        ax.plot(
            result.time,
            position,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8
        )

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Cart Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)

    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    return ax


def plot_tradeoff_scatter(
    results: List[SimulationResult],
    ax: Optional[plt.Axes] = None,
    title: str = "Performance Tradeoff: Settling Time vs Control Effort"
) -> plt.Axes:
    """Plot settling time vs control effort tradeoff.

    This visualization highlights the key engineering tradeoff between
    speed (settling time) and energy (control effort).

    Args:
        results: List of SimulationResult objects with metrics calculated
        ax: Optional matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Extract metrics
    for result in results:
        if not result.metrics:
            logger.warning(f"No metrics for {result.get_label()}, skipping")
            continue

        label = result.get_label()
        color, _, _ = _get_plot_style(label)

        settling_time = result.metrics['angle_settling_time']
        control_effort = result.metrics['control_effort_total']

        # Skip if didn't settle
        if settling_time >= np.inf:
            logger.warning(f"{label} did not settle, skipping from tradeoff plot")
            continue

        # Determine marker style based on controller type
        if 'LQR' in result.controller_type:
            marker = 'o'
            markersize = 12
        elif 'GAIN_SCHEDULED' in result.controller_type:
            marker = '^'
            markersize = 12
        elif 'CASCADE' in result.controller_type:
            marker = 's'
            markersize = 12
        else:  # PID
            marker = 'D'
            markersize = 10

        # Plot point
        ax.scatter(
            control_effort,
            settling_time,
            label=label,
            color=color,
            marker=marker,
            s=markersize**2,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )

        # Annotate point
        ax.annotate(
            label,
            xy=(control_effort, settling_time),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )

    ax.set_xlabel('Total Control Effort (N·s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Settling Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier annotation
    ax.text(
        0.02, 0.98,
        'Lower-left is better\n(Fast + Efficient)',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    return ax


def plot_noise_robustness(
    results: List[SimulationResult],
    ax: Optional[plt.Axes] = None,
    title: str = "Noise Robustness: Angle Error"
) -> plt.Axes:
    """Plot angle error comparison showing noise robustness.

    Args:
        results: List of SimulationResult objects
        ax: Optional matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        label = result.get_label()
        color, linestyle, linewidth = _get_plot_style(label)

        # Calculate absolute angle error
        angle_error = np.abs(result.states[:, 2])

        ax.plot(
            result.time,
            angle_error,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8
        )

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('|Angle Error| (rad)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')  # Log scale to see small errors
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', framealpha=0.9)

    return ax


def generate_comprehensive_report(
    results: List[SimulationResult],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Generate comprehensive 2x2 comparison figure.

    Creates a publication-quality figure with four subplots:
    - Top-left: Angle comparison
    - Top-right: Force comparison
    - Bottom-left: Position comparison
    - Bottom-right: Tradeoff scatter plot

    Args:
        results: List of SimulationResult objects with metrics calculated
        save_path: Optional path to save figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure object
    """
    logger.info("Generating comprehensive comparison report")

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'Inverted Pendulum Control: Comprehensive Performance Comparison',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Top-left: Angle comparison
    plot_angle_comparison(results, ax=axes[0, 0])

    # Top-right: Force comparison
    plot_force_comparison(results, ax=axes[0, 1])

    # Bottom-left: Position comparison
    plot_position_comparison(results, ax=axes[1, 0])

    # Bottom-right: Tradeoff scatter
    plot_tradeoff_scatter(results, ax=axes[1, 1])

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_kalman_filter_performance(
    result: SimulationResult,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot Kalman filter estimation performance.

    Shows true state vs measured vs estimated for scenarios with noise and filter.

    Args:
        result: SimulationResult with estimated_states
        save_path: Optional path to save figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure object or None if no filter data
    """
    if not result.use_filter or result.estimated_states is None:
        logger.warning("No Kalman filter data to plot")
        return None

    logger.info(f"Plotting Kalman filter performance for {result.get_label()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Kalman Filter Performance: {result.get_label()}',
        fontsize=16,
        fontweight='bold'
    )

    state_names = ['Position (m)', 'Velocity (m/s)', 'Angle (rad)', 'Angular Velocity (rad/s)']

    # Downsample for clarity (plot every Nth point)
    plot_step = max(1, len(result.time) // 1000)

    for i in range(4):
        ax = axes[i // 2, i % 2]

        # True state
        ax.plot(
            result.time[::plot_step],
            result.states[::plot_step, i],
            'g-',
            label='True State',
            linewidth=2,
            alpha=0.7
        )

        # Estimated state (downsampled to match control rate)
        control_steps = len(result.estimated_states)
        est_time = np.linspace(0, result.time[-1], control_steps)
        ax.plot(
            est_time,
            result.estimated_states[:, i],
            'b-',
            label='Kalman Estimate',
            linewidth=1.5,
            alpha=0.8
        )

        # Noisy measurements (only for position and angle)
        if result.measurements is not None and i in [0, 2]:
            ax.plot(
                est_time,
                result.measurements[:, i],
                'r.',
                label='Noisy Measurement',
                markersize=2,
                alpha=0.3
            )

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(state_names[i], fontsize=11)
        ax.set_title(f'{state_names[i]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Kalman filter plot saved to {save_path}")

    if show:
        plt.show()

    return fig
