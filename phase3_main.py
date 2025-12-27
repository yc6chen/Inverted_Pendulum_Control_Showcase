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
    CascadePIDController,
    GainScheduledCascadePID,
    linearize_pendulum
)


def create_controller(
    config: SimulationConfig,
    controller_type: str
) -> tuple:
    """Create a controller based on the specified type.

    Args:
        config: Simulation configuration
        controller_type: 'PID', 'CASCADE_PID', 'GAIN_SCHEDULED_PID', or 'LQR'

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

    elif controller_type == 'CASCADE_PID':
        # Cascade PID with inner angle loop and outer position loop
        # Tuning strategy:
        # - Inner loop stabilizes angle (must be fast and responsive)
        # - Outer loop slowly adjusts angle setpoint to control position (must be gentle)
        controller = CascadePIDController(
            angle_pid_gains=(100.0, 1.0, 30.0),    # Inner loop: strong angle control
            position_pid_gains=(0.5, 0.0, 2.0),    # Outer loop: gentle position correction
            dt=config.simulation.dt_control,
            saturation_limits=config.control.saturation_limits
        )
        print("\nCascade PID Controller Configuration:")
        print("-" * 60)
        print("Inner Loop (Angle Control):")
        print(f"  Kp={controller.angle_pid.Kp}, Ki={controller.angle_pid.Ki}, "
              f"Kd={controller.angle_pid.Kd}")
        print("Outer Loop (Position Control):")
        print(f"  Kp={controller.position_pid.Kp}, Ki={controller.position_pid.Ki}, "
              f"Kd={controller.position_pid.Kd}")
        print("  Strategy: Inner loop stabilizes angle, outer loop gently adjusts")
        print("  angle setpoint to bring cart back to center")
        print("-" * 60)
        return controller, "Cascade PID"

    elif controller_type == 'GAIN_SCHEDULED_PID':
        # Gain-scheduled cascade PID with adaptive gains
        controller = GainScheduledCascadePID(
            dt=config.simulation.dt_control,
            saturation_limits=config.control.saturation_limits
        )
        print("\nGain-Scheduled Cascade PID Controller Configuration:")
        print("-" * 60)
        print("Operating Modes:")
        print("  EMERGENCY (|θ| > 15°):")
        print(f"    Angle PID: {controller.gain_sets['emergency']['angle']}")
        print(f"    Position PID: {controller.gain_sets['emergency']['position']}")
        print("  STABILIZING (5° < |θ| < 15°):")
        print(f"    Angle PID: {controller.gain_sets['stabilizing']['angle']}")
        print(f"    Position PID: {controller.gain_sets['stabilizing']['position']}")
        print("  FINE-TUNING (|θ| < 5°):")
        print(f"    Angle PID: {controller.gain_sets['fine_tuning']['angle']}")
        print(f"    Position PID: {controller.gain_sets['fine_tuning']['position']}")
        print("  Strategy: Adapts gains based on pendulum angle magnitude")
        print("-" * 60)
        return controller, "Gain-Scheduled PID"

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
    cascade_config: SimulationConfig,
    lqr_config: SimulationConfig
) -> tuple:
    """Run simulations with PID, Cascade PID, and LQR controllers.

    Args:
        pid_config: Configuration for original PID controller
        cascade_config: Configuration for cascade PID controller
        lqr_config: Configuration for LQR controller

    Returns:
        Tuple of (pid_data, cascade_data, lqr_data, controllers_dict)
    """
    # Create system instances
    pid_system = CartPendulumSystem(pid_config.physical)
    cascade_system = CartPendulumSystem(cascade_config.physical)
    lqr_system = CartPendulumSystem(lqr_config.physical)

    # Create controllers
    pid_controller, _ = create_controller(pid_config, 'PID')
    cascade_controller, _ = create_controller(cascade_config, 'CASCADE_PID')
    lqr_controller, _ = create_controller(lqr_config, 'LQR')

    # Define controller functions for simulation
    def pid_control_fn(t, state):
        theta = state[2]  # Pendulum angle only
        return pid_controller.compute(pid_config.control.setpoint, theta)

    def cascade_control_fn(t, state):
        return cascade_controller.compute(state)  # Uses full state

    def lqr_control_fn(t, state):
        return lqr_controller.compute(state)

    # Run simulations
    print("\nRunning Original PID simulation...")
    pid_logger = SimulationLogger()
    pid_data = run_simulation(
        pid_system,
        pid_control_fn,
        pid_config.simulation,
        pid_logger
    )

    print("Running Cascade PID simulation...")
    cascade_logger = SimulationLogger()
    cascade_data = run_simulation(
        cascade_system,
        cascade_control_fn,
        cascade_config.simulation,
        cascade_logger
    )

    print("Running LQR simulation...")
    lqr_logger = SimulationLogger()
    lqr_data = run_simulation(
        lqr_system,
        lqr_control_fn,
        lqr_config.simulation,
        lqr_logger
    )

    controllers = {
        'PID': pid_controller,
        'Cascade PID': cascade_controller,
        'LQR': lqr_controller
    }

    return pid_data, cascade_data, lqr_data, controllers


def plot_comparison(pid_data: dict, cascade_data: dict, lqr_data: dict) -> None:
    """Plot comparison of PID, Cascade PID, and LQR controller performance.

    Args:
        pid_data: Simulation data from original PID controller
        cascade_data: Simulation data from cascade PID controller
        lqr_data: Simulation data from LQR controller
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 3: Controller Comparison (PID vs Cascade PID vs LQR)',
                 fontsize=16, fontweight='bold')

    # Extract data
    pid_time = pid_data['time']
    pid_states = pid_data['states']
    pid_forces = pid_data['forces']

    cascade_time = cascade_data['time']
    cascade_states = cascade_data['states']
    cascade_forces = cascade_data['forces']

    lqr_time = lqr_data['time']
    lqr_states = lqr_data['states']
    lqr_forces = lqr_data['forces']

    # Plot 1: Pendulum Angle
    ax1 = axes[0, 0]
    ax1.plot(pid_time, np.rad2deg(pid_states[:, 2]), 'b-', label='PID (angle only)',
             linewidth=1.5, alpha=0.7)
    ax1.plot(cascade_time, np.rad2deg(cascade_states[:, 2]), 'g-', label='Cascade PID',
             linewidth=1.5, alpha=0.8)
    ax1.plot(lqr_time, np.rad2deg(lqr_states[:, 2]), 'r-', label='LQR',
             linewidth=1.5, alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pendulum Angle (deg)')
    ax1.set_title('Pendulum Angle Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-50, 50])

    # Plot 2: Cart Position
    ax2 = axes[0, 1]
    ax2.plot(pid_time, pid_states[:, 0], 'b-', label='PID (angle only)',
             linewidth=1.5, alpha=0.7)
    ax2.plot(cascade_time, cascade_states[:, 0], 'g-', label='Cascade PID',
             linewidth=1.5, alpha=0.8)
    ax2.plot(lqr_time, lqr_states[:, 0], 'r-', label='LQR',
             linewidth=1.5, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cart Position (m)')
    ax2.set_title('Cart Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control Force
    ax3 = axes[1, 0]
    ax3.plot(pid_time, pid_forces, 'b-', label='PID (angle only)',
             linewidth=1.5, alpha=0.7)
    ax3.plot(cascade_time, cascade_forces, 'g-', label='Cascade PID',
             linewidth=1.5, alpha=0.8)
    ax3.plot(lqr_time, lqr_forces, 'r-', label='LQR',
             linewidth=1.5, alpha=0.8)
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
             'b-', label='PID (angle only)', linewidth=1.5, alpha=0.5)
    ax4.plot(np.rad2deg(cascade_states[:, 2]), np.rad2deg(cascade_states[:, 3]),
             'g-', label='Cascade PID', linewidth=1.5, alpha=0.7)
    ax4.plot(np.rad2deg(lqr_states[:, 2]), np.rad2deg(lqr_states[:, 3]),
             'r-', label='LQR', linewidth=1.5, alpha=0.7)
    ax4.scatter([np.rad2deg(pid_states[0, 2])], [np.rad2deg(pid_states[0, 3])],
                c='black', s=100, marker='o', label='Start', zorder=5)
    ax4.scatter([0], [0], c='gold', s=150, marker='*', label='Target', zorder=5)
    ax4.set_xlabel('Pendulum Angle (deg)')
    ax4.set_ylabel('Angular Velocity (deg/s)')
    ax4.set_title('Phase Portrait')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-30, 30])
    ax4.set_ylim([-100, 100])

    plt.tight_layout()
    plt.savefig('phase3_controller_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'phase3_controller_comparison.png'")
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


def print_performance_comparison(pid_metrics: dict, cascade_metrics: dict, lqr_metrics: dict) -> None:
    """Print performance metrics comparison table.

    Args:
        pid_metrics: Original PID controller metrics
        cascade_metrics: Cascade PID controller metrics
        lqr_metrics: LQR controller metrics
    """
    print("\n" + "=" * 90)
    print("PERFORMANCE COMPARISON: PID vs Cascade PID vs LQR")
    print("=" * 90)
    print(f"{'Metric':<30} {'PID (angle)':>18} {'Cascade PID':>18} {'LQR':>18}")
    print("-" * 90)

    # Settling Time
    print(f"{'Settling Time (s)':<30} {pid_metrics['settling_time']:>18.3f} "
          f"{cascade_metrics['settling_time']:>18.3f} {lqr_metrics['settling_time']:>18.3f}")

    # Control Effort
    print(f"{'Control Effort (N·s)':<30} {pid_metrics['control_effort']:>18.2f} "
          f"{cascade_metrics['control_effort']:>18.2f} {lqr_metrics['control_effort']:>18.2f}")

    # Peak Overshoot
    print(f"{'Peak Overshoot (deg)':<30} {pid_metrics['peak_overshoot']:>18.3f} "
          f"{cascade_metrics['peak_overshoot']:>18.3f} {lqr_metrics['peak_overshoot']:>18.3f}")

    # Steady-State Error
    print(f"{'Steady-State Error (deg)':<30} {pid_metrics['steady_state_error']:>18.4f} "
          f"{cascade_metrics['steady_state_error']:>18.4f} {lqr_metrics['steady_state_error']:>18.4f}")

    print("=" * 90)

    # Print improvement summary
    print("\nKey Observations:")
    print("-" * 90)

    # Check if PID is stable
    if pid_metrics['settling_time'] > 9.5:
        print("• Original PID: FAILS - pendulum oscillates indefinitely, cart runs away")
        print("  - Only controls angle, ignoring cart position")
    else:
        print("• Original PID: Stabilizes but with poor performance")

    # Cascade PID vs PID
    if cascade_metrics['settling_time'] < pid_metrics['settling_time']:
        cascade_improvement = ((pid_metrics['settling_time'] - cascade_metrics['settling_time']) /
                              pid_metrics['settling_time'] * 100)
        print(f"• Cascade PID: {cascade_improvement:.1f}% faster settling than original PID")
        print("  - Uses two-loop architecture to control both angle and position")

    # LQR vs Cascade PID
    if lqr_metrics['settling_time'] < cascade_metrics['settling_time']:
        lqr_improvement = ((cascade_metrics['settling_time'] - lqr_metrics['settling_time']) /
                           cascade_metrics['settling_time'] * 100)
        print(f"• LQR: {lqr_improvement:.1f}% faster settling than Cascade PID")
        print("  - Optimal full-state feedback with explicit system model")

    # Control effort comparison
    print(f"\n• Control Effort:")
    print(f"  - PID uses {pid_metrics['control_effort']:.1f} N·s")
    print(f"  - Cascade PID uses {cascade_metrics['control_effort']:.1f} N·s "
          f"({100 * cascade_metrics['control_effort'] / pid_metrics['control_effort']:.1f}% of PID)")
    print(f"  - LQR uses {lqr_metrics['control_effort']:.1f} N·s "
          f"({100 * lqr_metrics['control_effort'] / pid_metrics['control_effort']:.1f}% of PID)")

    print("=" * 90)


def run_comparison_simulation_extended(
    pid_config: SimulationConfig,
    cascade_config: SimulationConfig,
    gain_scheduled_config: SimulationConfig,
    lqr_config: SimulationConfig
) -> dict:
    """Run simulations with all four controllers.

    Args:
        pid_config: Configuration for original PID controller
        cascade_config: Configuration for cascade PID controller
        gain_scheduled_config: Configuration for gain-scheduled PID
        lqr_config: Configuration for LQR controller

    Returns:
        Dictionary mapping controller names to simulation data
    """
    results = {}

    # Create systems and controllers
    configs = {
        'PID': (pid_config, 'PID'),
        'Cascade PID': (cascade_config, 'CASCADE_PID'),
        'Gain-Scheduled PID': (gain_scheduled_config, 'GAIN_SCHEDULED_PID'),
        'LQR': (lqr_config, 'LQR')
    }

    for name, (config, controller_type) in configs.items():
        print(f"\nRunning {name} simulation...")
        system = CartPendulumSystem(config.physical)
        controller, _ = create_controller(config, controller_type)

        # Define controller function
        if controller_type == 'PID':
            def control_fn(t, state):
                return controller.compute(config.control.setpoint, state[2])
        else:
            def control_fn(t, state):
                return controller.compute(state)

        # Run simulation
        logger = SimulationLogger()
        data = run_simulation(system, control_fn, config.simulation, logger)
        results[name] = data

    return results


def print_performance_comparison_extended(metrics: dict) -> None:
    """Print extended performance metrics comparison table.

    Args:
        metrics: Dictionary mapping controller names to metrics
    """
    print("\n" + "=" * 110)
    print("PERFORMANCE COMPARISON: All Controllers")
    print("=" * 110)

    controllers = ['PID', 'Cascade PID', 'Gain-Scheduled PID', 'LQR']
    print(f"{'Metric':<30} {'PID':>18} {'Cascade':>18} {'Gain-Sched':>18} {'LQR':>18}")
    print("-" * 110)

    # Settling Time
    print(f"{'Settling Time (s)':<30} ", end='')
    for ctrl in controllers:
        print(f"{metrics[ctrl]['settling_time']:>18.3f} ", end='')
    print()

    # Control Effort
    print(f"{'Control Effort (N·s)':<30} ", end='')
    for ctrl in controllers:
        print(f"{metrics[ctrl]['control_effort']:>18.2f} ", end='')
    print()

    # Peak Overshoot
    print(f"{'Peak Overshoot (deg)':<30} ", end='')
    for ctrl in controllers:
        print(f"{metrics[ctrl]['peak_overshoot']:>18.3f} ", end='')
    print()

    # Steady-State Error
    print(f"{'Steady-State Error (deg)':<30} ", end='')
    for ctrl in controllers:
        print(f"{metrics[ctrl]['steady_state_error']:>18.4f} ", end='')
    print()

    print("=" * 110)

    # Analysis
    print("\nKey Observations:")
    print("-" * 110)

    # Check gain-scheduled performance
    gs_metrics = metrics['Gain-Scheduled PID']
    if gs_metrics['settling_time'] < 15.0 and gs_metrics['steady_state_error'] < 10.0:
        print("✓ Gain-Scheduled PID: SUCCEEDS - Adapts gains for stable control!")
        print("  - Switches between emergency, stabilizing, and fine-tuning modes")
        print("  - Successfully balances pendulum and controls cart position")
    elif gs_metrics['settling_time'] < metrics['Cascade PID']['settling_time']:
        print("○ Gain-Scheduled PID: Partial success - Better than fixed-gain cascade")
        print(f"  - {100*(1 - gs_metrics['settling_time']/metrics['Cascade PID']['settling_time']):.1f}% faster than Cascade PID")
    else:
        print("✗ Gain-Scheduled PID: Needs further tuning")

    # Compare to LQR
    if gs_metrics['settling_time'] > metrics['LQR']['settling_time']:
        improvement = 100 * (1 - metrics['LQR']['settling_time'] / gs_metrics['settling_time'])
        print(f"\n• LQR still {improvement:.1f}% faster than best PID variant")
        print("  - Demonstrates superiority of model-based optimal control")

    print("=" * 110)


def plot_comparison_extended(results: dict, metrics: dict) -> None:
    """Plot extended comparison of all controllers.

    Args:
        results: Dictionary of simulation data
        metrics: Dictionary of performance metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Extended Controller Comparison (20s simulation)',
                 fontsize=16, fontweight='bold')

    colors = {
        'PID': '#1f77b4',                # Blue
        'Cascade PID': '#ff7f0e',        # Orange
        'Gain-Scheduled PID': '#2ca02c', # Green
        'LQR': '#d62728'                 # Red
    }

    # Plot 1: Pendulum Angle
    ax1 = axes[0, 0]
    for name, data in results.items():
        stable = metrics[name]['steady_state_error'] < 10.0
        label = f"{name} {'✓' if stable else '✗'}"
        ax1.plot(data['time'], np.rad2deg(data['states'][:, 2]),
                color=colors[name], label=label, linewidth=1.5, alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pendulum Angle (deg)')
    ax1.set_title('Pendulum Angle Response')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-30, 30])

    # Plot 2: Cart Position
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(data['time'], data['states'][:, 0],
                color=colors[name], label=name, linewidth=1.5, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cart Position (m)')
    ax2.set_title('Cart Position')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control Force
    ax3 = axes[0, 2]
    for name, data in results.items():
        ax3.plot(data['time'], data['forces'],
                color=colors[name], label=name, linewidth=1.5, alpha=0.8)
    ax3.axhline(y=20, color='k', linestyle='--', alpha=0.3, label='Limits')
    ax3.axhline(y=-20, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Force (N)')
    ax3.set_title('Control Effort')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Angle - First 10 seconds zoom
    ax4 = axes[1, 0]
    for name, data in results.items():
        mask = data['time'] <= 10.0
        ax4.plot(data['time'][mask], np.rad2deg(data['states'][mask, 2]),
                color=colors[name], label=name, linewidth=1.5, alpha=0.8)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pendulum Angle (deg)')
    ax4.set_title('Angle Response (0-10s zoom)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Performance metrics bar chart
    ax5 = axes[1, 1]
    controllers = list(results.keys())
    x = np.arange(len(controllers))
    settling_times = [metrics[c]['settling_time'] for c in controllers]
    bars = ax5.bar(x, settling_times, color=[colors[c] for c in controllers])
    ax5.set_ylabel('Settling Time (s)')
    ax5.set_title('Settling Time Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(controllers, rotation=15, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, settling_times):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

    # Plot 6: Phase portrait
    ax6 = axes[1, 2]
    for name, data in results.items():
        states = data['states']
        ax6.plot(np.rad2deg(states[:, 2]), np.rad2deg(states[:, 3]),
                color=colors[name], label=name, linewidth=1.5, alpha=0.7)
    ax6.scatter([np.rad2deg(results['PID']['states'][0, 2])],
               [np.rad2deg(results['PID']['states'][0, 3])],
               c='black', s=100, marker='o', label='Start', zorder=5)
    ax6.scatter([0], [0], c='gold', s=150, marker='*', label='Target', zorder=5)
    ax6.set_xlabel('Pendulum Angle (deg)')
    ax6.set_ylabel('Angular Velocity (deg/s)')
    ax6.set_title('Phase Portrait')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([-30, 30])
    ax6.set_ylim([-150, 150])

    plt.tight_layout()
    plt.savefig('phase3_extended_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'phase3_extended_comparison.png'")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 100)
    print("PHASE 3: EXTENDED CONTROLLER COMPARISON")
    print("PID vs Cascade PID vs Gain-Scheduled PID vs LQR")
    print("=" * 100)

    # Extended simulation time for PID controllers
    sim_time = 20.0  # Extended from 10s to 20s to give PID more time

    # Create configuration for original PID controller
    pid_config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(sim_time=sim_time),
        control=ControlParameters(
            controller_type='PID',
            pid_gains=PIDGains(Kp=50.0, Ki=30.0, Kd=15.0)
        )
    )

    # Create configuration for cascade PID controller
    cascade_config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(sim_time=sim_time),
        control=ControlParameters(
            controller_type='CASCADE_PID'
        )
    )

    # Create configuration for gain-scheduled PID controller
    gain_scheduled_config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(sim_time=sim_time),
        control=ControlParameters(
            controller_type='GAIN_SCHEDULED_PID'
        )
    )

    # Create configuration for LQR controller
    lqr_config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(sim_time=sim_time),
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
    print("-" * 100)
    print(f"Physical Parameters: M={pid_config.physical.M} kg, "
          f"m={pid_config.physical.m} kg, l={pid_config.physical.l} m")
    print(f"Simulation Duration: {sim_time} s (Extended to give PID more time)")
    print(f"Initial Angle: {np.rad2deg(pid_config.simulation.initial_state[2]):.2f} deg")

    # Run comparison
    results = run_comparison_simulation_extended(
        pid_config,
        cascade_config,
        gain_scheduled_config,
        lqr_config
    )

    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = {name: calculate_metrics(data) for name, data in results.items()}

    # Print comparison
    print_performance_comparison_extended(metrics)

    # Plot results
    print("\nGenerating comparison plots...")
    plot_comparison_extended(results, metrics)

    print("\nPhase 3 extended demonstration complete!")


if __name__ == "__main__":
    main()
