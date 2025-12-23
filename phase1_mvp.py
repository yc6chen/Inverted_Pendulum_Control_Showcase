"""
Phase 1: MVP - Physics Engine & Open-Loop Simulation

This script implements the non-linear dynamics of an inverted pendulum on a cart
without any control input. The pendulum starts from a small angle (0.1 rad) and
naturally falls due to gravity, demonstrating the inherent instability of the system.

Key Components:
- Non-linear cart-pole equations of motion derived from Lagrangian mechanics
- High-fidelity simulation using scipy.integrate.solve_ivp (RK45 method)
- Real-time animation showing cart movement and pendulum rotation
- Time-series plot of pendulum angle evolution

Physical System:
- Cart mass (M): 1.0 kg
- Pendulum mass (m): 0.3 kg
- Pendulum length (l): 0.5 m (pivot to center of gravity)
- Gravity (g): 9.81 m/s²
- Cart friction (b): 0.1 N/m/s

State Vector: [x, x_dot, theta, theta_dot]
where x = cart position, theta = angle from upright (0 = upright)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp


# System Parameters
M = 1.0         # Cart mass (kg)
m = 0.3         # Pendulum mass (kg)
l = 0.5         # Pendulum length to CoG (m)
g = 9.81        # Gravity (m/s²)
b = 0.1         # Cart friction coefficient (N/m/s)
dt = 0.001      # Simulation timestep (s)
sim_time = 3.0  # Total simulation time (s)


def state_derivatives(t, state, force=0.0):
    """
    Compute state derivatives for the cart-pole system.

    Implements non-linear equations of motion:
    x_ddot = (F + m*l*theta_dot² *sin(theta) - b*x_dot - m*g*cos(theta)*sin(theta)) / denom
    theta_ddot = (g*sin(theta) - cos(theta)*x_ddot) / l

    where denom = M + m - m*cos²(theta)

    Args:
        t: Current time (s)
        state: [x, x_dot, theta, theta_dot]
        force: Control force applied to cart (N) - set to 0 for Phase 1

    Returns:
        derivatives: [x_dot, x_ddot, theta_dot, theta_ddot]
    """
    x, x_dot, theta, theta_dot = state

    # Precompute trigonometric values
    S = np.sin(theta)
    C = np.cos(theta)

    total_mass = M + m
    denom = total_mass - m * C**2

    # Cart acceleration
    x_ddot = (force + m*l*theta_dot**2*S - b*x_dot - m*g*C*S) / denom

    # Pendulum angular acceleration
    theta_ddot = (g*S - C*x_ddot) / l

    return [x_dot, x_ddot, theta_dot, theta_ddot]


def simulate_pendulum():
    """
    Simulate the cart-pole system with no control input.

    Returns:
        t: Time array (s)
        states: Solution array [N x 4] containing state history
    """
    # Initial condition: pendulum starts at 0.1 rad (~5.7°) from upright
    initial_state = [0.0, 0.0, 0.1, 0.0]  # [x, x_dot, theta, theta_dot]

    # Time span for integration
    t_span = (0, sim_time)
    t_eval = np.arange(0, sim_time, dt)

    # Solve ODE using RK45 method
    solution = solve_ivp(
        state_derivatives,
        t_span,
        initial_state,
        method='RK45',
        t_eval=t_eval,
        args=(0.0,)  # No control force
    )

    return solution.t, solution.y.T


def create_animation(t, states):
    """
    Create real-time animation of the cart-pole system.

    Args:
        t: Time array
        states: State history [N x 4]
    """
    fig, (ax_anim, ax_plot) = plt.subplots(1, 2, figsize=(14, 5))

    # Animation subplot setup
    ax_anim.set_xlim(-2, 2)
    ax_anim.set_ylim(-1, 1)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, alpha=0.3)
    ax_anim.set_xlabel('Position (m)')
    ax_anim.set_ylabel('Height (m)')
    ax_anim.set_title('Cart-Pole System (No Control)')

    # Reference vertical line (target upright position)
    ax_anim.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Target')

    # Initialize cart (rectangle)
    cart_width = 0.3
    cart_height = 0.15
    cart = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height,
                         fill=True, color='blue', ec='black')
    ax_anim.add_patch(cart)

    # Initialize pendulum (line + bob)
    pendulum_line, = ax_anim.plot([], [], 'o-', lw=3, markersize=12,
                                   color='red', label='Pendulum')

    # Time text
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)
    ax_anim.legend(loc='upper right')

    # Plot subplot: theta vs time
    ax_plot.plot(t, states[:, 2], 'b-', linewidth=2)
    ax_plot.set_xlabel('Time (s)')
    ax_plot.set_ylabel('Pendulum Angle θ (rad)')
    ax_plot.set_title('Angle Evolution (Open Loop)')
    ax_plot.grid(True, alpha=0.3)
    ax_plot.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Upright')

    # Current time indicator (vertical line)
    time_line = ax_plot.axvline(x=0, color='red', linestyle='-', alpha=0.7)
    ax_plot.legend()

    def init():
        """Initialize animation."""
        pendulum_line.set_data([], [])
        return pendulum_line, cart, time_text, time_line

    def animate(i):
        """Update animation frame."""
        # Get current state
        x = states[i, 0]
        theta = states[i, 2]

        # Update cart position
        cart.set_x(x - cart_width/2)

        # Update pendulum position
        # Pendulum bob position: cart_pos + l*sin(theta) in x, l*cos(theta) in y
        pendulum_x = [x, x + l * np.sin(theta)]
        pendulum_y = [0, l * np.cos(theta)]
        pendulum_line.set_data(pendulum_x, pendulum_y)

        # Update time text
        time_text.set_text(f'Time: {t[i]:.2f} s')

        # Update time indicator on plot
        time_line.set_xdata([t[i], t[i]])

        return pendulum_line, cart, time_text, time_line

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=dt*1000, blit=True, repeat=True
    )

    plt.tight_layout()
    return anim


def main():
    """Main execution function."""
    print("=" * 60)
    print("Phase 1: Inverted Pendulum MVP - Open Loop Simulation")
    print("=" * 60)
    print(f"\nSystem Parameters:")
    print(f"  Cart mass (M):        {M} kg")
    print(f"  Pendulum mass (m):    {m} kg")
    print(f"  Pendulum length (l):  {l} m")
    print(f"  Gravity (g):          {g} m/s²")
    print(f"  Friction (b):         {b} N/m/s")
    print(f"  Simulation time:      {sim_time} s")
    print(f"  Timestep (dt):        {dt} s")
    print(f"\nInitial Condition: θ₀ = 0.1 rad (~5.7°)")
    print(f"\nSimulating unstable system (no control)...\n")

    # Run simulation
    t, states = simulate_pendulum()

    # Display final state
    print(f"Final State:")
    print(f"  Cart position:      {states[-1, 0]:.3f} m")
    print(f"  Cart velocity:      {states[-1, 1]:.3f} m/s")
    print(f"  Pendulum angle:     {states[-1, 2]:.3f} rad ({np.degrees(states[-1, 2]):.1f}°)")
    print(f"  Angular velocity:   {states[-1, 3]:.3f} rad/s")
    print(f"\nGenerating animation and plot...")

    # Create and display animation
    anim = create_animation(t, states)
    plt.show()

    print("\n" + "=" * 60)
    print("Simulation complete. The pendulum falls as expected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
