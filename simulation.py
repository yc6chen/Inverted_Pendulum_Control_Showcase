"""
Simulation Module for Inverted Pendulum System

This module implements the physics engine for the cart-pendulum system,
including the non-linear equations of motion and ODE integration.

The physics are based on Lagrangian mechanics and account for:
- Cart mass and friction
- Pendulum mass distribution
- Gravitational effects
- Control force input
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, Dict
from config import PhysicalParameters, SimulationParameters


class CartPendulumSystem:
    """Physics simulation for an inverted pendulum on a cart.

    This class encapsulates the non-linear dynamics of the cart-pole system
    and provides methods for state evolution and integration.

    Attributes:
        params: Physical parameters of the system
        state: Current state vector [x, x_dot, theta, theta_dot]
    """

    def __init__(self, params: PhysicalParameters):
        """Initialize the cart-pendulum system.

        Args:
            params: Physical parameters (masses, length, friction, gravity)
        """
        self.params = params
        self.state = np.array([0.0, 0.0, 0.0, 0.0])

    def state_derivatives(
        self, t: float, state: np.ndarray, force: float = 0.0
    ) -> np.ndarray:
        """Compute state derivatives for the cart-pole system.

        Implements the non-linear equations of motion derived from
        Lagrangian mechanics:

        x_ddot = (F + m*l*theta_dot²*sin(θ) - b*x_dot - m*g*cos(θ)*sin(θ)) / denom
        theta_ddot = (g*sin(θ) - cos(θ)*x_ddot) / l

        where denom = M + m - m*cos²(θ)

        Args:
            t: Current time (s) - not used but required by ODE solver
            state: State vector [x, x_dot, theta, theta_dot]
            force: Control force applied to cart (N)

        Returns:
            State derivatives [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x, x_dot, theta, theta_dot = state

        # Extract parameters
        M = self.params.M
        m = self.params.m
        l = self.params.l
        b = self.params.b
        g = self.params.g

        # Precompute trigonometric values
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Denominator in equations of motion
        total_mass = M + m
        denom = total_mass - m * cos_theta**2

        # Cart acceleration
        numerator_x = (
            force
            + m * l * theta_dot**2 * sin_theta
            - b * x_dot
            - m * g * cos_theta * sin_theta
        )
        x_ddot = numerator_x / denom

        # Pendulum angular acceleration
        theta_ddot = (g * sin_theta - cos_theta * x_ddot) / l

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def step(self, force: float, dt: float) -> np.ndarray:
        """Advance simulation by one timestep using RK45 integration.

        Args:
            force: Control force to apply (N)
            dt: Timestep duration (s)

        Returns:
            Updated state vector [x, x_dot, theta, theta_dot]
        """
        # Integrate from current state for duration dt
        solution = solve_ivp(
            fun=lambda t, y: self.state_derivatives(t, y, force),
            t_span=(0, dt),
            y0=self.state,
            method='RK45',
            dense_output=True
        )

        # Update internal state
        self.state = solution.y[:, -1]
        return self.state.copy()

    def reset(self, initial_state: Tuple[float, float, float, float]) -> None:
        """Reset the system to a specified initial state.

        Args:
            initial_state: Initial state [x, x_dot, theta, theta_dot]
        """
        self.state = np.array(initial_state)

    def get_state(self) -> np.ndarray:
        """Get the current state of the system.

        Returns:
            Current state vector [x, x_dot, theta, theta_dot]
        """
        return self.state.copy()


class SimulationLogger:
    """Data logger for simulation runs.

    Records time-series data during simulation for later analysis and plotting.

    Attributes:
        time: List of timestamps
        states: List of state vectors
        forces: List of control forces
        measurements: List of measured states (if applicable)
    """

    def __init__(self):
        """Initialize empty log arrays."""
        self.time = []
        self.states = []
        self.forces = []
        self.measurements = []

    def log_step(
        self,
        t: float,
        state: np.ndarray,
        force: float,
        measurement: np.ndarray = None
    ) -> None:
        """Log data from a single simulation step.

        Args:
            t: Current time (s)
            state: Current state vector
            force: Applied control force (N)
            measurement: Measured state (optional, for noisy scenarios)
        """
        self.time.append(t)
        self.states.append(state.copy())
        self.forces.append(force)
        if measurement is not None:
            self.measurements.append(measurement.copy())

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Convert logged data to numpy arrays.

        Returns:
            Dictionary containing:
                - 'time': Time array (N,)
                - 'states': State array (N, 4)
                - 'forces': Force array (N,)
                - 'measurements': Measurement array (N, 4) if available
        """
        data = {
            'time': np.array(self.time),
            'states': np.array(self.states),
            'forces': np.array(self.forces)
        }

        if self.measurements:
            data['measurements'] = np.array(self.measurements)

        return data

    def clear(self) -> None:
        """Clear all logged data."""
        self.time.clear()
        self.states.clear()
        self.forces.clear()
        self.measurements.clear()


def run_simulation(
    system: CartPendulumSystem,
    controller: Callable[[float, np.ndarray], float],
    sim_params: SimulationParameters,
    logger: SimulationLogger = None
) -> Dict[str, np.ndarray]:
    """Run a complete simulation with a given controller.

    The simulation runs at high frequency (dt_plant) for accurate physics,
    but the controller only updates at a slower rate (dt_control) to
    simulate realistic control loop timing.

    Args:
        system: CartPendulumSystem instance
        controller: Function that takes (time, state) and returns control force
        sim_params: Simulation parameters (timesteps, duration, initial state)
        logger: Optional logger instance (creates new one if None)

    Returns:
        Dictionary of logged data arrays
    """
    if logger is None:
        logger = SimulationLogger()
    else:
        logger.clear()

    # Reset system to initial state
    system.reset(sim_params.initial_state)

    # Calculate number of steps and control update interval
    num_steps = int(sim_params.sim_time / sim_params.dt_plant)
    control_steps_per_update = int(sim_params.dt_control / sim_params.dt_plant)

    # Initialize control force
    force = 0.0

    # Simulation loop
    for step in range(num_steps):
        t = step * sim_params.dt_plant
        current_state = system.get_state()

        # Update control force only at control intervals
        if step % control_steps_per_update == 0:
            force = controller(t, current_state)

        # Log before stepping
        logger.log_step(t, current_state, force)

        # Step physics forward
        system.step(force, sim_params.dt_plant)

    return logger.get_arrays()
