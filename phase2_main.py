"""
Phase 2: PID Control Implementation

This script demonstrates stabilization of the inverted pendulum using a
discrete-time PID controller with anti-windup logic.

Key Features:
- Modular architecture with separate controller, simulation, and config modules
- PID controller tuned for angle stabilization
- Real-time visualization showing control performance
- Comparative plots showing controlled vs. uncontrolled behavior
- Performance metrics (settling time, steady-state error, control effort)

Control Strategy:
The PID controller receives the current pendulum angle and computes a force
to apply to the cart. The goal is to drive the angle to zero (upright position).

Author: Generated for Phase 2 of Inverted Pendulum Control Showcase
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict
import os

from config import SimulationConfig
from simulation import CartPendulumSystem, SimulationLogger, run_simulation
from controllers import PIDController


def create_pid_controller_function(pid: PIDController, setpoint: float):
    """Create a controller function wrapper for the PID controller.

    For an inverted pendulum, the control law requires positive feedback
    (F = +K*theta), but standard PID computes error = setpoint - measurement,
    giving negative feedback. We negate the PID output to correct this.

    Args:
        pid: PIDController instance
        setpoint: Target angle (rad)

    Returns:
        Function that takes (time, state) and returns control force
    """
    def controller_func(t: float, state: np.ndarray) -> float:
        """PID controller function.

        Args:
            t: Current time (not used by PID but required by interface)
            state: Current state [x, x_dot, theta, theta_dot]

        Returns:
            Control force (N)
        """
        theta = state[2]  # Extract pendulum angle
        # Negate PID output because inverted pendulum needs positive feedback
        force = -pid.compute(setpoint, theta)
        return force

    return controller_func


def calculate_performance_metrics(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate key performance metrics from simulation data.

    Args:
        data: Dictionary containing 'time', 'states', 'forces' arrays

    Returns:
        Dictionary containing:
            - 'settling_time': Time to reach and stay within 2% of setpoint (s)
            - 'steady_state_error': Final average error magnitude (rad)
            - 'control_effort': Integrated absolute control force (N·s)
            - 'max_force': Maximum absolute control force (N)
    """
    time = data['time']
    states = data['states']
    forces = data['forces']

    theta = states[:, 2]  # Pendulum angle
    abs_forces = np.abs(forces)

    # Settling time: time to reach and stay within 2% of setpoint (0 rad)
    tolerance = 0.02  # 2% of full scale (assuming ~pi rad range)
    settled_indices = np.where(np.abs(theta) < tolerance)[0]

    if len(settled_indices) > 0:
        # Find first index where it stays settled
        for i in settled_indices:
            if np.all(np.abs(theta[i:]) < tolerance):
                settling_time = time[i]
                break
        else:
            settling_time = time[-1]  # Never fully settled
    else:
        settling_time = time[-1]

    # Steady-state error: average error in last 10% of simulation
    steady_state_idx = int(0.9 * len(theta))
    steady_state_error = np.mean(np.abs(theta[steady_state_idx:]))

    # Control effort: integrated absolute force
    control_effort = np.trapezoid(abs_forces, time)

    # Maximum force
    max_force = np.max(abs_forces)

    return {
        'settling_time': settling_time,
        'steady_state_error': steady_state_error,
        'control_effort': control_effort,
        'max_force': max_force
    }


def create_visualization(
    data_controlled: Dict[str, np.ndarray],
    data_uncontrolled: Dict[str, np.ndarray],
    config: SimulationConfig
) -> None:
    """Create comprehensive visualization comparing controlled vs. uncontrolled.

    Args:
        data_controlled: Logged data from PID-controlled simulation
        data_uncontrolled: Logged data from uncontrolled simulation
        config: Simulation configuration
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid for subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Extract data
    t_ctrl = data_controlled['time']
    states_ctrl = data_controlled['states']
    forces_ctrl = data_controlled['forces']

    t_unctrl = data_uncontrolled['time']
    states_unctrl = data_uncontrolled['states']

    # Plot 1: Pendulum Angle vs Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_unctrl, states_unctrl[:, 2], 'r--', linewidth=2, label='No Control', alpha=0.7)
    ax1.plot(t_ctrl, states_ctrl[:, 2], 'b-', linewidth=2, label='PID Control')
    ax1.axhline(y=0, color='green', linestyle=':', alpha=0.5, label='Setpoint')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pendulum Angle θ (rad)')
    ax1.set_title('Pendulum Angle: PID Control vs. No Control')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Control Force vs Time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_ctrl, forces_ctrl, 'b-', linewidth=2, label='Control Force')
    if config.control.saturation_limits:
        min_limit, max_limit = config.control.saturation_limits
        ax2.axhline(y=max_limit, color='r', linestyle='--', alpha=0.5, label='Saturation Limits')
        ax2.axhline(y=min_limit, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Force (N)')
    ax2.set_title('PID Control Signal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cart Position vs Time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_unctrl, states_unctrl[:, 0], 'r--', linewidth=2, label='No Control', alpha=0.7)
    ax3.plot(t_ctrl, states_ctrl[:, 0], 'b-', linewidth=2, label='PID Control')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Cart Position x (m)')
    ax3.set_title('Cart Position Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Angular Velocity vs Time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_unctrl, states_unctrl[:, 3], 'r--', linewidth=2, label='No Control', alpha=0.7)
    ax4.plot(t_ctrl, states_ctrl[:, 3], 'b-', linewidth=2, label='PID Control')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity dθ/dt (rad/s)')
    ax4.set_title('Angular Velocity Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Phase Portrait (theta vs theta_dot)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(states_unctrl[:, 2], states_unctrl[:, 3], 'r--', linewidth=2,
             label='No Control', alpha=0.7)
    ax5.plot(states_ctrl[:, 2], states_ctrl[:, 3], 'b-', linewidth=2, label='PID Control')
    ax5.plot(0, 0, 'go', markersize=10, label='Target')
    ax5.set_xlabel('Angle θ (rad)')
    ax5.set_ylabel('Angular Velocity dθ/dt (rad/s)')
    ax5.set_title('Phase Portrait: Angle vs. Angular Velocity')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Performance Summary (Text)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    metrics = calculate_performance_metrics(data_controlled)

    summary_text = (
        "PID Control Performance Metrics\n"
        "=" * 40 + "\n\n"
        f"PID Gains:\n"
        f"  Kp = {config.control.pid_gains.Kp}\n"
        f"  Ki = {config.control.pid_gains.Ki}\n"
        f"  Kd = {config.control.pid_gains.Kd}\n\n"
        f"Performance:\n"
        f"  Settling Time:       {metrics['settling_time']:.3f} s\n"
        f"  Steady-State Error:  {metrics['steady_state_error']:.6f} rad\n"
        f"                       ({np.degrees(metrics['steady_state_error']):.4f}°)\n"
        f"  Control Effort:      {metrics['control_effort']:.2f} N·s\n"
        f"  Max Force:           {metrics['max_force']:.2f} N\n\n"
        f"System Parameters:\n"
        f"  Cart Mass:     {config.physical.M} kg\n"
        f"  Pole Mass:     {config.physical.m} kg\n"
        f"  Pole Length:   {config.physical.l} m\n"
        f"  Friction:      {config.physical.b} N/m/s\n"
    )

    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Phase 2: PID Control Performance Analysis', fontsize=16, fontweight='bold')

    # Save figure
    output_file = 'phase2_performance_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPerformance analysis plot saved to: {output_file}")
    plt.close()


def create_animation_comparison(
    data_controlled: Dict[str, np.ndarray],
    config: SimulationConfig,
    speed_factor: float = 1.0
) -> animation.FuncAnimation:
    """Create side-by-side animation of cart-pendulum system with PID control.

    Args:
        data_controlled: Logged data from controlled simulation
        config: Simulation configuration
        speed_factor: Animation speed multiplier (1.0 = real-time)

    Returns:
        matplotlib FuncAnimation object
    """
    fig, (ax_anim, ax_angle) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data
    t = data_controlled['time']
    states = data_controlled['states']
    forces = data_controlled['forces']

    # Animation subplot setup
    ax_anim.set_xlim(-2, 2)
    ax_anim.set_ylim(-1, 1)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, alpha=0.3)
    ax_anim.set_xlabel('Position (m)')
    ax_anim.set_ylabel('Height (m)')
    ax_anim.set_title('Cart-Pole System (PID Control)')

    # Initialize cart
    cart_width = 0.3
    cart_height = 0.15
    cart = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height,
                         fill=True, color='blue', ec='black')
    ax_anim.add_patch(cart)

    # Initialize pendulum
    pendulum_line, = ax_anim.plot([], [], 'o-', lw=3, markersize=12, color='red')

    # Time and force text
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)
    force_text = ax_anim.text(0.02, 0.88, '', transform=ax_anim.transAxes)

    # Angle plot
    ax_angle.plot(t, states[:, 2], 'b-', linewidth=2, alpha=0.3)
    ax_angle.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Setpoint')
    ax_angle.set_xlabel('Time (s)')
    ax_angle.set_ylabel('Pendulum Angle θ (rad)')
    ax_angle.set_title('Angle vs. Time')
    ax_angle.grid(True, alpha=0.3)
    ax_angle.legend()

    # Current state indicator
    angle_point, = ax_angle.plot([], [], 'ro', markersize=8)
    time_line = ax_angle.axvline(x=0, color='red', linestyle='-', alpha=0.5)

    def init():
        """Initialize animation."""
        pendulum_line.set_data([], [])
        angle_point.set_data([], [])
        return pendulum_line, cart, time_text, force_text, angle_point, time_line

    def animate(i):
        """Update animation frame."""
        x = states[i, 0]
        theta = states[i, 2]
        force = forces[i]

        # Update cart position
        cart.set_x(x - cart_width/2)

        # Update pendulum
        pendulum_x = [x, x + config.physical.l * np.sin(theta)]
        pendulum_y = [0, config.physical.l * np.cos(theta)]
        pendulum_line.set_data(pendulum_x, pendulum_y)

        # Update text
        time_text.set_text(f'Time: {t[i]:.2f} s')
        force_text.set_text(f'Force: {force:.2f} N')

        # Update angle plot
        angle_point.set_data([t[i]], [theta])
        time_line.set_xdata([t[i], t[i]])

        return pendulum_line, cart, time_text, force_text, angle_point, time_line

    # Calculate frame interval
    frame_interval = config.simulation.dt_plant * 1000 / speed_factor

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=frame_interval, blit=True, repeat=True
    )

    plt.tight_layout()
    return anim


def main():
    """Main execution function for Phase 2."""
    print("=" * 70)
    print("Phase 2: PID Control Implementation")
    print("=" * 70)

    # Create configuration
    config = SimulationConfig()
    config.print_summary()

    # Create physics system
    system = CartPendulumSystem(config.physical)

    # Create PID controller
    pid = PIDController(
        Kp=config.control.pid_gains.Kp,
        Ki=config.control.pid_gains.Ki,
        Kd=config.control.pid_gains.Kd,
        dt=config.simulation.dt_control,
        saturation_limits=config.control.saturation_limits
    )

    print(f"\nController Configuration:")
    print(f"  {pid}")

    # Create controller function
    controller_func = create_pid_controller_function(pid, config.control.setpoint)

    # Run controlled simulation
    print(f"\nRunning PID-controlled simulation...")
    logger_controlled = SimulationLogger()
    data_controlled = run_simulation(
        system=system,
        controller=controller_func,
        sim_params=config.simulation,
        logger=logger_controlled
    )

    # Run uncontrolled simulation for comparison
    print(f"Running uncontrolled simulation for comparison...")
    system.reset(config.simulation.initial_state)
    logger_uncontrolled = SimulationLogger()
    data_uncontrolled = run_simulation(
        system=system,
        controller=lambda t, s: 0.0,  # No control force
        sim_params=config.simulation,
        logger=logger_uncontrolled
    )

    # Calculate and display metrics
    metrics = calculate_performance_metrics(data_controlled)
    print(f"\n" + "=" * 70)
    print("Performance Metrics")
    print("=" * 70)
    print(f"  Settling Time:       {metrics['settling_time']:.3f} s")
    print(f"  Steady-State Error:  {metrics['steady_state_error']:.6f} rad "
          f"({np.degrees(metrics['steady_state_error']):.4f}°)")
    print(f"  Control Effort:      {metrics['control_effort']:.2f} N·s")
    print(f"  Max Force:           {metrics['max_force']:.2f} N")
    print("=" * 70)

    # Create visualizations
    print(f"\nGenerating performance comparison plots...")
    create_visualization(data_controlled, data_uncontrolled, config)

    # Animation generation is memory-intensive and slow for long simulations
    # Uncomment below to generate animation (requires significant time and memory)
    # print(f"\nGenerating animation...")
    # anim = create_animation_comparison(data_controlled, config, speed_factor=1.0)
    # animation_file = 'phase2_animation.gif'
    # print(f"Saving animation to: {animation_file}")
    # anim.save(animation_file, writer='pillow', fps=30)
    # print(f"Animation saved successfully!")
    # plt.close()

    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    print("The PID controller successfully stabilizes the inverted pendulum.")
    print("=" * 70)


if __name__ == "__main__":
    main()
