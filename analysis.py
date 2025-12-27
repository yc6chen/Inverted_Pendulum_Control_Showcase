"""
Analysis Module for Inverted Pendulum Control System

This module provides a structured pipeline for running simulations and analyzing
controller performance. It implements Phase 5's requirements for reproducibility
and systematic data management.

The module includes:
- SimulationResult: Structured dataclass for simulation outputs
- SimulationRunner: Orchestrates simulation execution with proper logging
- Helper utilities for reproducible analysis
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import subprocess

from config import SimulationConfig, PhysicalParameters
from simulation import CartPendulumSystem, SimulationLogger
from controllers import PIDController, LQRController, linearize_pendulum, CascadePIDController, GainScheduledCascadePID
from filters import KalmanFilter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Structured container for simulation results and metadata.

    This dataclass ensures consistent data structure and enables easy
    comparison between different simulation runs.

    Attributes:
        config: The configuration used for this simulation
        time: Time array (N,)
        states: State array (N, 4) - [x, x_dot, theta, theta_dot]
        forces: Control force array (N,)
        measurements: Measured states if noise was applied (N, 4)
        estimated_states: Kalman filter estimates if used (N, 4)
        controller_type: Type of controller ('PID', 'LQR', etc.)
        use_noise: Whether sensor noise was applied
        use_filter: Whether Kalman filter was used
        timestamp: When the simulation was run
        git_hash: Git commit hash for reproducibility
        metrics: Performance metrics (computed separately)
    """
    config: SimulationConfig
    time: np.ndarray
    states: np.ndarray
    forces: np.ndarray
    measurements: Optional[np.ndarray] = None
    estimated_states: Optional[np.ndarray] = None
    controller_type: str = ""
    use_noise: bool = False
    use_filter: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    git_hash: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Automatically populate git hash for reproducibility."""
        if not self.git_hash:
            try:
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    self.git_hash = result.stdout.strip()[:8]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.git_hash = "unknown"

    def get_label(self) -> str:
        """Generate a human-readable label for this simulation.

        Returns:
            Label string like 'PID-Ideal' or 'LQR-Noisy+KF'
        """
        noise_str = "Noisy" if self.use_noise else "Ideal"
        filter_str = "+KF" if self.use_filter else ""
        return f"{self.controller_type}-{noise_str}{filter_str}"


class SimulationRunner:
    """Orchestrates simulation execution with proper data flow and logging.

    This class takes a configuration, runs the simulation, and returns
    a structured SimulationResult. It handles:
    - Controller initialization
    - Optional noise injection
    - Optional Kalman filter state estimation
    - Comprehensive logging

    Attributes:
        config: Simulation configuration
        use_noise: Whether to add sensor noise
        use_filter: Whether to use Kalman filter
        noise_std: Standard deviation for sensor noise
    """

    def __init__(
        self,
        config: SimulationConfig,
        use_noise: bool = False,
        use_filter: bool = False,
        noise_std: Tuple[float, float] = (0.01, 0.005)
    ):
        """Initialize the simulation runner.

        Args:
            config: Complete simulation configuration
            use_noise: Add Gaussian noise to measurements
            use_filter: Use Kalman filter for state estimation
            noise_std: (position_std, angle_std) for noise generation
        """
        self.config = config
        self.use_noise = use_noise
        self.use_filter = use_filter
        self.noise_std = noise_std

        # Validate configuration
        if use_filter and not use_noise:
            logger.warning(
                "Kalman filter enabled without noise. "
                "Filter will have minimal effect on ideal measurements."
            )

    def _create_controller(self):
        """Create the appropriate controller based on configuration.

        Returns:
            Tuple of (controller_object, controller_function)
            where controller_function takes (time, state) -> force
        """
        controller_type = self.config.control.controller_type
        logger.info(f"Initializing {controller_type} controller")

        if controller_type == 'PID':
            pid = PIDController(
                Kp=self.config.control.pid_gains.Kp,
                Ki=self.config.control.pid_gains.Ki,
                Kd=self.config.control.pid_gains.Kd,
                dt=self.config.simulation.dt_control,
                saturation_limits=self.config.control.saturation_limits
            )

            def controller_func(t: float, state: np.ndarray) -> float:
                """PID control based on angle error."""
                theta = state[2]
                return pid.compute(self.config.control.setpoint, theta)

            return pid, controller_func

        elif controller_type == 'CASCADE_PID':
            cascade_pid = CascadePIDController(
                angle_pid_gains=(120.0, 2.0, 35.0),
                position_pid_gains=(1.2, 0.05, 4.0),
                dt=self.config.simulation.dt_control,
                saturation_limits=self.config.control.saturation_limits
            )

            def controller_func(t: float, state: np.ndarray) -> float:
                """Cascade PID control."""
                return cascade_pid.compute(state, position_setpoint=0.0)

            return cascade_pid, controller_func

        elif controller_type == 'GAIN_SCHEDULED_PID':
            gs_pid = GainScheduledCascadePID(
                dt=self.config.simulation.dt_control,
                saturation_limits=self.config.control.saturation_limits
            )

            def controller_func(t: float, state: np.ndarray) -> float:
                """Gain-scheduled cascade PID control."""
                return gs_pid.compute(state, position_setpoint=0.0)

            return gs_pid, controller_func

        elif controller_type == 'LQR':
            # Linearize system
            A, B = linearize_pendulum(
                M=self.config.physical.M,
                m=self.config.physical.m,
                l=self.config.physical.l,
                b=self.config.physical.b,
                g=self.config.physical.g
            )

            # Create LQR controller
            lqr = LQRController(
                A=A,
                B=B,
                Q=self.config.control.lqr_gains.get_Q_matrix(),
                R=self.config.control.lqr_gains.get_R_matrix(),
                saturation_limits=self.config.control.saturation_limits
            )

            logger.info(f"LQR gain matrix K: {lqr.K}")
            logger.info(f"Closed-loop eigenvalues: {lqr.E}")

            def controller_func(t: float, state: np.ndarray) -> float:
                """LQR state-feedback control."""
                return lqr.compute(state)

            return lqr, controller_func

        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

    def _create_kalman_filter(self) -> Optional[KalmanFilter]:
        """Create Kalman filter if needed.

        Returns:
            KalmanFilter instance or None
        """
        if not self.use_filter:
            return None

        logger.info("Initializing Kalman Filter")

        # Get linearized system matrices
        A, B = linearize_pendulum(
            M=self.config.physical.M,
            m=self.config.physical.m,
            l=self.config.physical.l,
            b=self.config.physical.b,
            g=self.config.physical.g
        )

        # Discretize using zero-order hold
        dt = self.config.simulation.dt_control
        A_d = np.eye(4) + A * dt
        B_d = B * dt

        # Measurement matrix - we measure position and angle only
        C_d = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Measure x
            [0.0, 0.0, 1.0, 0.0]   # Measure theta
        ])

        # Process noise covariance (model uncertainty)
        Q_kf = np.eye(4) * 1e-4

        # Measurement noise covariance
        R_kf = np.diag([self.noise_std[0]**2, self.noise_std[1]**2])

        # Initial state estimate
        x0 = np.array(self.config.simulation.initial_state)

        kf = KalmanFilter(
            A_d=A_d,
            B_d=B_d,
            C_d=C_d,
            Q=Q_kf,
            R=R_kf,
            x0=x0
        )

        return kf

    def _add_noise(self, state: np.ndarray) -> np.ndarray:
        """Add sensor noise to position and angle measurements.

        Args:
            state: True state [x, x_dot, theta, theta_dot]

        Returns:
            Noisy measurement [x_noisy, x_dot, theta_noisy, theta_dot]
        """
        if not self.use_noise:
            return state.copy()

        noisy_state = state.copy()
        # Add noise to position (index 0)
        noisy_state[0] += np.random.randn() * self.noise_std[0]
        # Add noise to angle (index 2)
        noisy_state[2] += np.random.randn() * self.noise_std[1]

        return noisy_state

    def run(self) -> SimulationResult:
        """Run the simulation and return structured results.

        Returns:
            SimulationResult containing all data and metadata
        """
        logger.info("=" * 60)
        logger.info(f"Starting simulation: {self.config.control.controller_type}")
        logger.info(f"  Noise: {self.use_noise}, Filter: {self.use_filter}")
        logger.info("=" * 60)

        # Create system
        system = CartPendulumSystem(self.config.physical)
        system.reset(self.config.simulation.initial_state)

        # Create controller
        controller_obj, controller_func = self._create_controller()

        # Create Kalman filter if needed
        kf = self._create_kalman_filter()

        # Initialize logging
        logger_sim = SimulationLogger()

        # Lists for storing data
        estimated_states_list = [] if self.use_filter else None
        measurements_list = [] if self.use_noise else None

        # Simulation parameters
        num_steps = int(self.config.simulation.sim_time / self.config.simulation.dt_plant)
        control_steps_per_update = self.config.simulation.control_steps_per_sim_step

        # Initialize control
        force = 0.0
        estimated_state = np.array(self.config.simulation.initial_state)

        # Main simulation loop
        for step in range(num_steps):
            t = step * self.config.simulation.dt_plant
            true_state = system.get_state()

            # Update control at control intervals
            if step % control_steps_per_update == 0:
                # Get measurement (potentially noisy)
                measured_state = self._add_noise(true_state)

                # Update Kalman filter if enabled
                if kf is not None:
                    # Predict step
                    kf.predict(np.array([force]))
                    # Update step with measurement
                    measurement_vector = np.array([measured_state[0], measured_state[2]])
                    kf.update(measurement_vector)
                    estimated_state = kf.get_state_estimate()
                    estimated_states_list.append(estimated_state.copy())
                else:
                    estimated_state = measured_state

                # Store measurements if noise is applied
                if self.use_noise:
                    measurements_list.append(measured_state.copy())

                # Compute control using estimated state
                force = controller_func(t, estimated_state)

            # Log data
            logger_sim.log_step(t, true_state, force)

            # Step physics forward
            system.step(force, self.config.simulation.dt_plant)

        # Get logged arrays
        data = logger_sim.get_arrays()

        logger.info(f"Simulation completed: {num_steps} steps")
        logger.info("=" * 60)

        # Create result object
        result = SimulationResult(
            config=self.config,
            time=data['time'],
            states=data['states'],
            forces=data['forces'],
            measurements=np.array(measurements_list) if measurements_list else None,
            estimated_states=np.array(estimated_states_list) if estimated_states_list else None,
            controller_type=self.config.control.controller_type,
            use_noise=self.use_noise,
            use_filter=self.use_filter
        )

        return result
