"""
Phase 3: LQR Control Demonstration

This script demonstrates the Linear Quadratic Regulator (LQR) controller
for stabilizing the inverted pendulum system. It compares the performance
of the optimal LQR controller against the classical PID controller from Phase 2.

Key Features:
- System linearization around the unstable equilibrium point
- LQR optimal gain computation using the Algebraic Riccati Equation
- Configuration-driven controller selection
- Performance comparison between PID and LQR controllers
- Visualization of control performance and system response

The LQR controller is model-based and provides optimal control by minimizing
a quadratic cost function that balances state errors and control effort.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import (
    SimulationConfig,
    PhysicalParameters,
    SimulationParameters,
    ControlParameters,
    PIDGains,
    LQRGains
)
from simulation import CartPendulumSystem, SimulationLogger, run_simulation
from controllers import (
    PIDController,
    LQRController,
    linearize_pendulum
)


def create_controller(
    config: SimulationConfig,
    controller_type: str
) -> tuple:
    """Create a controller based on the specified type.

    Args:
        config: Simulation configuration
        controller_type: 'PID' or 'LQR'

    Returns:
        Tuple of (controller_object, controller_name)
    """
    if controller_type == 'PID':
        controller = PIDController(
            Kp=config.control.pid_gains.Kp,
            Ki=config.control.pid_gains.Ki,
            Kd=config.control.pid_gains.Kd,
            dt=config.simulation.dt_control,
            saturation_limits=config.control.saturation_limits
        )
        return controller, "PID"

    elif controller_type == 'LQR':
        # Linearize the system around upright equilibrium
        A, B = linearize_pendulum(
            M=config.physical.M,
            m=config.physical.m,
            l=config.physical.l,
            b=config.physical.b,
            g=config.physical.g
        )

        # Get cost matrices
        Q = config.control.lqr_gains.get_Q_matrix()
        R = config.control.lqr_gains.get_R_matrix()

        # Create LQR controller
        controller = LQRController(
            A=A,
            B=B,
            Q=Q,
            R=R,
            saturation_limits=config.control.saturation_limits
        )

        # Print LQR gains and eigenvalues
        print("\nLQR Controller Design:")
        print("-" * 60)
        print("Gain Matrix K:")
        print(f"  {controller.get_gain_matrix()}")
        print("\nClosed-Loop Eigenvalues:")
        eigenvalues = controller.get_closed_loop_eigenvalues()
        for i, eig in enumerate(eigenvalues):
            print(f"  λ{i+1} = {eig:.4f}")
        print("-" * 60)

        return controller, "LQR"

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def run_comparison_simulation(
    pid_config: SimulationConfig,
    lqr_config: SimulationConfig
) -> tuple:
    """Run simulations with both PID and LQR controllers.

    Args:
        pid_config: Configuration for PID controller
        lqr_config: Configuration for LQR controller

    Returns:
        Tuple of (pid_data, lqr_data, pid_controller, lqr_controller)
    """
    # Create system instances
    pid_system = CartPendulumSystem(pid_config.physical)
    lqr_system = CartPendulumSystem(lqr_config.physical)

    # Create controllers
    pid_controller, _ = create_controller(pid_config, 'PID')
    lqr_controller, _ = create_controller(lqr_config, 'LQR')

    # Define controller functions for simulation
    def pid_control_fn(t, state):
        theta = state[2]  # Pendulum angle
        return pid_controller.compute(pid_config.control.setpoint, theta)

    def lqr_control_fn(t, state):
        return lqr_controller.compute(state)

    # Run simulations
    print("\nRunning PID simulation...")
    pid_logger = SimulationLogger()
    pid_data = run_simulation(
        pid_system,
        pid_control_fn,
        pid_config.simulation,
        pid_logger
    )

    print("Running LQR simulation...")
    lqr_logger = SimulationLogger()
    lqr_data = run_simulation(
        lqr_system,
        lqr_control_fn,
        lqr_config.simulation,
        lqr_logger
    )

    return pid_data, lqr_data, pid_controller, lqr_controller


def plot_comparison(pid_data: dict, lqr_data: dict) -> None:
    """Plot comparison of PID and LQR controller performance.

    Args:
        pid_data: Simulation data from PID controller
        lqr_data: Simulation data from LQR controller
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 3: PID vs LQR Control Comparison', fontsize=16, fontweight='bold')

    # Extract data
    pid_time = pid_data['time']
    pid_states = pid_data['states']
    pid_forces = pid_data['forces']

    lqr_time = lqr_data['time']
    lqr_states = lqr_data['states']
    lqr_forces = lqr_data['forces']

    # Plot 1: Pendulum Angle
    ax1 = axes[0, 0]
    ax1.plot(pid_time, np.rad2deg(pid_states[:, 2]), 'b-', label='PID', linewidth=1.5)
    ax1.plot(lqr_time, np.rad2deg(lqr_states[:, 2]), 'r-', label='LQR', linewidth=1.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pendulum Angle (deg)')
    ax1.set_title('Pendulum Angle Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cart Position
    ax2 = axes[0, 1]
    ax2.plot(pid_time, pid_states[:, 0], 'b-', label='PID', linewidth=1.5)
    ax2.plot(lqr_time, lqr_states[:, 0], 'r-', label='LQR', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cart Position (m)')
    ax2.set_title('Cart Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control Force
    ax3 = axes[1, 0]
    ax3.plot(pid_time, pid_forces, 'b-', label='PID', linewidth=1.5)
    ax3.plot(lqr_time, lqr_forces, 'r-', label='LQR', linewidth=1.5)
    ax3.axhline(y=20, color='k', linestyle='--', alpha=0.3, label='Saturation Limits')
    ax3.axhline(y=-20, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Force (N)')
    ax3.set_title('Control Effort')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Phase Portrait (Angle vs Angular Velocity)
    ax4 = axes[1, 1]
    ax4.plot(np.rad2deg(pid_states[:, 2]), np.rad2deg(pid_states[:, 3]),
             'b-', label='PID', linewidth=1.5, alpha=0.7)
    ax4.plot(np.rad2deg(lqr_states[:, 2]), np.rad2deg(lqr_states[:, 3]),
             'r-', label='LQR', linewidth=1.5, alpha=0.7)
    ax4.scatter([np.rad2deg(pid_states[0, 2])], [np.rad2deg(pid_states[0, 3])],
                c='blue', s=100, marker='o', label='Start', zorder=5)
    ax4.scatter([0], [0], c='green', s=100, marker='*', label='Target', zorder=5)
    ax4.set_xlabel('Pendulum Angle (deg)')
    ax4.set_ylabel('Angular Velocity (deg/s)')
    ax4.set_title('Phase Portrait')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase3_lqr_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'phase3_lqr_comparison.png'")
    plt.show()


def calculate_metrics(data: dict) -> dict:
    """Calculate performance metrics from simulation data.

    Args:
        data: Simulation data dictionary

    Returns:
        Dictionary of performance metrics
    """
    time = data['time']
    states = data['states']
    forces = data['forces']

    # Settling time (time to reach and stay within 2% of final value)
    theta = states[:, 2]
    final_value = theta[-1]
    threshold = 0.02 * abs(theta[0] - final_value)

    settling_idx = len(theta) - 1
    for i in range(len(theta) - 1, -1, -1):
        if abs(theta[i] - final_value) > threshold:
            settling_idx = i
            break

    settling_time = time[settling_idx] if settling_idx < len(time) else time[-1]

    # Control effort (integrated absolute force)
    dt = time[1] - time[0] if len(time) > 1 else 0.01
    control_effort = np.sum(np.abs(forces)) * dt

    # Peak overshoot
    peak_overshoot = np.max(np.abs(theta))

    # Steady-state error
    steady_state_error = abs(theta[-1])

    return {
        'settling_time': settling_time,
        'control_effort': control_effort,
        'peak_overshoot': np.rad2deg(peak_overshoot),
        'steady_state_error': np.rad2deg(steady_state_error)
    }


def print_performance_comparison(pid_metrics: dict, lqr_metrics: dict) -> None:
    """Print performance metrics comparison table.

    Args:
        pid_metrics: PID controller metrics
        lqr_metrics: LQR controller metrics
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: PID vs LQR")
    print("=" * 70)
    print(f"{'Metric':<30} {'PID':>15} {'LQR':>15} {'Improvement':>10}")
    print("-" * 70)

    # Settling Time
    st_improvement = ((pid_metrics['settling_time'] - lqr_metrics['settling_time']) /
                      pid_metrics['settling_time'] * 100)
    print(f"{'Settling Time (s)':<30} {pid_metrics['settling_time']:>15.3f} "
          f"{lqr_metrics['settling_time']:>15.3f} {st_improvement:>9.1f}%")

    # Control Effort
    ce_improvement = ((pid_metrics['control_effort'] - lqr_metrics['control_effort']) /
                      pid_metrics['control_effort'] * 100)
    print(f"{'Control Effort (N·s)':<30} {pid_metrics['control_effort']:>15.2f} "
          f"{lqr_metrics['control_effort']:>15.2f} {ce_improvement:>9.1f}%")

    # Peak Overshoot
    po_improvement = ((pid_metrics['peak_overshoot'] - lqr_metrics['peak_overshoot']) /
                      pid_metrics['peak_overshoot'] * 100)
    print(f"{'Peak Overshoot (deg)':<30} {pid_metrics['peak_overshoot']:>15.3f} "
          f"{lqr_metrics['peak_overshoot']:>15.3f} {po_improvement:>9.1f}%")

    # Steady-State Error
    print(f"{'Steady-State Error (deg)':<30} {pid_metrics['steady_state_error']:>15.4f} "
          f"{lqr_metrics['steady_state_error']:>15.4f}")

    print("=" * 70)
    print("\nNote: Positive improvement % means LQR performs better")
    print("=" * 70)


def main():
    """Main execution function."""
    print("=" * 70)
    print("PHASE 3: LINEAR QUADRATIC REGULATOR (LQR) CONTROL")
    print("=" * 70)

    # Create configuration for PID controller
    pid_config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(),
        control=ControlParameters(
            controller_type='PID',
            pid_gains=PIDGains(Kp=50.0, Ki=30.0, Kd=15.0)
        )
    )

    # Create configuration for LQR controller
    lqr_config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(),
        control=ControlParameters(
            controller_type='LQR',
            lqr_gains=LQRGains(
                Q=(1.0, 0.0, 10.0, 0.0),  # Penalize position and angle
                R=0.01  # Control effort weight
            )
        )
    )

    # Print configuration summary
    print("\nConfiguration Summary:")
    print("-" * 70)
    print(f"Physical Parameters: M={pid_config.physical.M} kg, "
          f"m={pid_config.physical.m} kg, l={pid_config.physical.l} m")
    print(f"Simulation Duration: {pid_config.simulation.sim_time} s")
    print(f"Initial Angle: {np.rad2deg(pid_config.simulation.initial_state[2]):.2f} deg")

    # Run comparison
    pid_data, lqr_data, pid_controller, lqr_controller = run_comparison_simulation(
        pid_config,
        lqr_config
    )

    # Calculate metrics
    print("\nCalculating performance metrics...")
    pid_metrics = calculate_metrics(pid_data)
    lqr_metrics = calculate_metrics(lqr_data)

    # Print comparison
    print_performance_comparison(pid_metrics, lqr_metrics)

    # Plot results
    print("\nGenerating comparison plots...")
    plot_comparison(pid_data, lqr_data)

    print("\nPhase 3 demonstration complete!")


if __name__ == "__main__":
    main()
