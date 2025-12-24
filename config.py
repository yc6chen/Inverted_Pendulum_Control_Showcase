"""
Configuration Module for Inverted Pendulum Control System

This module centralizes all system parameters, simulation settings, and controller
gains to avoid magic numbers and enable easy tuning.

The configuration is organized using dataclasses for type safety and clarity.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# Physical Constants
GRAVITY = 9.81  # Acceleration due to gravity (m/s²)


@dataclass
class PhysicalParameters:
    """Physical parameters of the cart-pendulum system.

    Attributes:
        M: Cart mass (kg)
        m: Pendulum mass (kg)
        l: Distance from pivot to pendulum center of mass (m)
        b: Cart friction coefficient (N/m/s)
        g: Gravitational acceleration (m/s²)
    """
    M: float = 1.0      # Cart mass (kg)
    m: float = 0.3      # Pendulum mass (kg)
    l: float = 0.5      # Pendulum length to CoG (m)
    b: float = 0.1      # Cart friction coefficient (N/m/s)
    g: float = GRAVITY  # Gravity (m/s²)


@dataclass
class SimulationParameters:
    """Simulation timing and numerical integration settings.

    Attributes:
        dt_plant: Physics simulation timestep (s) - high frequency for accuracy
        dt_control: Controller update interval (s) - realistic control loop rate
        sim_time: Total simulation duration (s)
        initial_state: Initial state vector [x, x_dot, theta, theta_dot]
    """
    dt_plant: float = 0.001      # Simulation timestep (s)
    dt_control: float = 0.01     # Control update interval (s)
    sim_time: float = 10.0       # Total simulation time (s)
    initial_state: Tuple[float, float, float, float] = (0.0, 0.0, 0.1, 0.0)

    @property
    def control_steps_per_sim_step(self) -> int:
        """Number of simulation steps per control update."""
        return int(self.dt_control / self.dt_plant)


@dataclass
class PIDGains:
    """PID controller gain parameters.

    Attributes:
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain
    """
    Kp: float = 50.0
    Ki: float = 30.0
    Kd: float = 15.0


@dataclass
class ControlParameters:
    """Control system configuration.

    Attributes:
        pid_gains: PID controller gains
        saturation_limits: Control force limits (N) as (min, max) tuple
        setpoint: Target angle (rad) - 0.0 for upright pendulum
    """
    pid_gains: PIDGains = None
    saturation_limits: Optional[Tuple[float, float]] = (-20.0, 20.0)
    setpoint: float = 0.0  # Target angle (rad)

    def __post_init__(self):
        """Initialize default PID gains if not provided."""
        if self.pid_gains is None:
            self.pid_gains = PIDGains()


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    This dataclass aggregates all configuration parameters for a simulation run.

    Attributes:
        physical: Physical system parameters
        simulation: Simulation timing parameters
        control: Control system parameters
    """
    physical: PhysicalParameters = None
    simulation: SimulationParameters = None
    control: ControlParameters = None

    def __post_init__(self):
        """Initialize default sub-configurations if not provided."""
        if self.physical is None:
            self.physical = PhysicalParameters()
        if self.simulation is None:
            self.simulation = SimulationParameters()
        if self.control is None:
            self.control = ControlParameters()

    def print_summary(self) -> None:
        """Print a formatted summary of all configuration parameters."""
        print("=" * 60)
        print("Simulation Configuration")
        print("=" * 60)
        print("\nPhysical Parameters:")
        print(f"  Cart mass (M):        {self.physical.M} kg")
        print(f"  Pendulum mass (m):    {self.physical.m} kg")
        print(f"  Pendulum length (l):  {self.physical.l} m")
        print(f"  Friction (b):         {self.physical.b} N/m/s")
        print(f"  Gravity (g):          {self.physical.g} m/s²")

        print("\nSimulation Parameters:")
        print(f"  Simulation timestep:  {self.simulation.dt_plant} s")
        print(f"  Control timestep:     {self.simulation.dt_control} s")
        print(f"  Simulation duration:  {self.simulation.sim_time} s")
        print(f"  Initial state:        {self.simulation.initial_state}")

        print("\nControl Parameters:")
        print(f"  PID Gains:")
        print(f"    Kp: {self.control.pid_gains.Kp}")
        print(f"    Ki: {self.control.pid_gains.Ki}")
        print(f"    Kd: {self.control.pid_gains.Kd}")
        print(f"  Saturation limits:    {self.control.saturation_limits} N")
        print(f"  Setpoint:             {self.control.setpoint} rad")
        print("=" * 60)
