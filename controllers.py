"""
Controllers Module for Inverted Pendulum System

This module implements various control strategies for stabilizing the inverted
pendulum system. Includes PID controller with anti-windup logic and LQR
(Linear Quadratic Regulator) optimal control.

The PID controller uses a discrete-time implementation suitable for real-time
control applications. The LQR controller uses state-space linearization around
the unstable equilibrium point.
"""

import numpy as np
from typing import Optional, Tuple
try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    print("Warning: 'control' library not available. LQR controller will not work.")
    print("Install with: pip install control")


class PIDController:
    """A discrete-time PID controller with integrator anti-windup.

    This controller implements the classic PID control algorithm:
        u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt

    The anti-windup mechanism prevents integrator saturation when the control
    output is limited, which is critical for real-world performance and stability.

    Attributes:
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain
        dt: Control timestep in seconds
        saturation_limits: Optional (min, max) output limits
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        dt: float,
        saturation_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """Initialize the PID controller.

        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            dt: Control timestep (s)
            saturation_limits: Optional (min, max) output limits for anti-windup
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.saturation_limits = saturation_limits

        # Internal state
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:
        """Compute the PID control signal.

        Calculates proportional, integral, and derivative terms based on the
        error between setpoint and measurement. Applies anti-windup logic to
        prevent integrator saturation.

        Args:
            setpoint: The desired target value (e.g., 0.0 rad for upright)
            measurement: The current measured value (e.g., current angle)

        Returns:
            The computed control output (saturated if limits are set)
        """
        # Calculate error
        error = setpoint - measurement

        # Proportional term
        p_term = self.Kp * error

        # Integral term (accumulated error)
        self._integral += error * self.dt
        i_term = self.Ki * self._integral

        # Derivative term (rate of change of error)
        derivative = (error - self._prev_error) / self.dt
        d_term = self.Kd * derivative

        # Compute raw output
        output = p_term + i_term + d_term

        # Apply saturation and anti-windup
        if self.saturation_limits is not None:
            min_limit, max_limit = self.saturation_limits

            # Check if we're saturating
            if output > max_limit:
                saturated_output = max_limit
                # Anti-windup: back-calculate integral to prevent further accumulation
                # Only apply if error and output have the same sign (not helping)
                if np.sign(error) == np.sign(output):
                    self._integral -= error * self.dt
            elif output < min_limit:
                saturated_output = min_limit
                # Anti-windup
                if np.sign(error) == np.sign(output):
                    self._integral -= error * self.dt
            else:
                saturated_output = output
        else:
            saturated_output = output

        # Store state for next iteration
        self._prev_error = error
        self._prev_output = saturated_output

        return saturated_output

    def reset(self) -> None:
        """Reset the controller's internal state.

        Clears the integral accumulator and previous error. Useful when
        restarting a simulation or switching control modes.
        """
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0

    def get_state(self) -> dict:
        """Get the current internal state of the controller.

        Returns:
            Dictionary containing:
                - 'integral': Current integral accumulator value
                - 'prev_error': Previous error value
                - 'prev_output': Previous control output
        """
        return {
            'integral': self._integral,
            'prev_error': self._prev_error,
            'prev_output': self._prev_output
        }

    def set_gains(self, Kp: float = None, Ki: float = None, Kd: float = None) -> None:
        """Update controller gains without resetting internal state.

        Args:
            Kp: New proportional gain (None to keep current)
            Ki: New integral gain (None to keep current)
            Kd: New derivative gain (None to keep current)
        """
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd

    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"PIDController(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, "
            f"dt={self.dt}, saturation_limits={self.saturation_limits})"
        )


def linearize_pendulum(M: float, m: float, l: float, b: float, g: float) -> Tuple[np.ndarray, np.ndarray]:
    """Linearize the cart-pendulum system around the upright equilibrium.

    Derives the state-space matrices A and B for the linearized system:
        dx/dt = A*x + B*u

    where the state vector is x = [x, x_dot, theta, theta_dot] and
    the control input is u = force.

    The linearization is performed around the unstable equilibrium point:
        x_eq = [0, 0, 0, 0] (cart at origin, pendulum upright, no velocity)

    Using small-angle approximations (sin(theta) ≈ theta, cos(theta) ≈ 1),
    the non-linear equations of motion become:

        x_ddot = (F - b*x_dot - m*g*theta) / M
        theta_ddot = (g*theta - x_ddot) / l

    Expanding theta_ddot and simplifying:
        theta_ddot = -F/(M*l) + b*x_dot/(M*l) + g*(M+m)*theta/(M*l)

    Args:
        M: Cart mass (kg)
        m: Pendulum mass (kg)
        l: Distance from pivot to pendulum center of mass (m)
        b: Cart friction coefficient (N/m/s)
        g: Gravitational acceleration (m/s²)

    Returns:
        Tuple of (A, B) where:
            A: State matrix (4x4)
            B: Input matrix (4x1)

    Raises:
        ValueError: If physical parameters are invalid (non-positive masses or length)
    """
    # Validate inputs
    if M <= 0 or m <= 0 or l <= 0:
        raise ValueError(
            f"Masses and length must be positive. Got M={M}, m={m}, l={l}"
        )

    # State matrix A (4x4)
    # State vector ordering: [x, x_dot, theta, theta_dot]
    A = np.array([
        [0.0,        1.0,                0.0,  0.0],
        [0.0,       -b/M,           -m*g/M,  0.0],
        [0.0,        0.0,                0.0,  1.0],
        [0.0,  b/(M*l),  g*(M + m)/(M*l),  0.0]
    ])

    # Input matrix B (4x1)
    B = np.array([
        [0.0],
        [1.0/M],
        [0.0],
        [-1.0/(M*l)]
    ])

    return A, B


def validate_lqr_matrices(Q: np.ndarray, R: np.ndarray) -> None:
    """Validate that Q and R matrices satisfy LQR requirements.

    Args:
        Q: State cost matrix (must be positive semi-definite)
        R: Control cost matrix (must be positive definite)

    Raises:
        ValueError: If matrices don't meet requirements
    """
    # Check Q is square
    if Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square. Got shape {Q.shape}")

    # Check Q is symmetric
    if not np.allclose(Q, Q.T):
        raise ValueError("Q must be symmetric")

    # Check Q is positive semi-definite (all eigenvalues >= 0)
    q_eigenvalues = np.linalg.eigvals(Q)
    if np.any(q_eigenvalues < -1e-10):  # Small tolerance for numerical errors
        raise ValueError(
            f"Q must be positive semi-definite. Got eigenvalues: {q_eigenvalues}"
        )

    # Check R is square
    if R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square. Got shape {R.shape}")

    # Check R is symmetric
    if not np.allclose(R, R.T):
        raise ValueError("R must be symmetric")

    # Check R is positive definite (all eigenvalues > 0)
    r_eigenvalues = np.linalg.eigvals(R)
    if np.any(r_eigenvalues <= 1e-10):
        raise ValueError(
            f"R must be positive definite. Got eigenvalues: {r_eigenvalues}"
        )


class LQRController:
    """Linear Quadratic Regulator (LQR) controller for the inverted pendulum.

    The LQR controller is an optimal state-feedback controller that minimizes
    a quadratic cost function:

        J = ∫ (x^T Q x + u^T R u) dt

    where:
        - x is the state error vector
        - u is the control input
        - Q penalizes state deviations
        - R penalizes control effort

    The optimal control law is: u = -K*x, where K is the gain matrix computed
    by solving the Algebraic Riccati Equation.

    Attributes:
        A: Linearized state matrix (4x4)
        B: Linearized input matrix (4x1)
        Q: State cost matrix (4x4)
        R: Control cost matrix (1x1)
        K: Computed LQR gain matrix (1x4)
        saturation_limits: Optional (min, max) output limits
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        saturation_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """Initialize the LQR controller.

        Validates the cost matrices and solves the LQR problem to compute
        the optimal gain matrix K.

        Args:
            A: Linearized state matrix from linearize_pendulum
            B: Linearized input matrix from linearize_pendulum
            Q: State cost matrix (4x4, positive semi-definite)
            R: Control cost matrix (1x1 or scalar, positive definite)
            saturation_limits: Optional (min, max) output limits for control signal

        Raises:
            ValueError: If cost matrices are invalid
            RuntimeError: If control library is not available or LQR solution fails
        """
        if not CONTROL_AVAILABLE:
            raise RuntimeError(
                "LQR controller requires 'control' library. "
                "Install with: pip install control"
            )

        self.A = A
        self.B = B
        self.saturation_limits = saturation_limits

        # Convert R to 2D array if scalar
        if np.isscalar(R):
            R = np.array([[R]])
        elif R.ndim == 1:
            R = R.reshape(-1, 1)

        self.Q = Q
        self.R = R

        # Validate cost matrices
        try:
            validate_lqr_matrices(Q, R)
        except ValueError as e:
            raise ValueError(f"Invalid cost matrices: {e}")

        # Solve LQR problem
        try:
            # Use continuous-time LQR
            K, S, E = control.lqr(A, B, Q, R)
            self.K = K
            self.S = S  # Solution to Riccati equation
            self.E = E  # Closed-loop eigenvalues

        except Exception as e:
            raise RuntimeError(
                f"Failed to solve LQR problem. The system may be uncontrollable. "
                f"Error: {e}"
            )

    def compute(self, state: np.ndarray) -> float:
        """Compute the LQR control signal.

        Implements the optimal control law: u = -K*x

        Args:
            state: Current state vector [x, x_dot, theta, theta_dot]

        Returns:
            Control force (N), saturated if limits are set
        """
        # Compute optimal control: u = -K*x
        # K has shape (1, 4), state has shape (4,), result is scalar
        control_output = -np.dot(self.K, state)[0]

        # Apply saturation if limits are set
        if self.saturation_limits is not None:
            min_limit, max_limit = self.saturation_limits
            control_output = np.clip(control_output, min_limit, max_limit)

        return control_output

    def get_gain_matrix(self) -> np.ndarray:
        """Get the computed LQR gain matrix.

        Returns:
            Gain matrix K (1x4)
        """
        return self.K.copy()

    def get_closed_loop_eigenvalues(self) -> np.ndarray:
        """Get the closed-loop eigenvalues of the system.

        The eigenvalues indicate the stability and response characteristics
        of the closed-loop system. All eigenvalues should have negative
        real parts for stability.

        Returns:
            Array of closed-loop eigenvalues
        """
        return self.E.copy()

    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"LQRController(K={self.K.flatten()}, "
            f"saturation_limits={self.saturation_limits})"
        )


class CascadePIDController:
    """Cascade PID controller for inverted pendulum system.

    This controller uses a two-loop architecture to address the fundamental
    limitation of single-variable PID control:

    - Inner Loop (Fast): Stabilizes the pendulum angle
    - Outer Loop (Slow): Controls cart position by adjusting angle setpoint

    This architecture mimics human balancing behavior: to keep a stick upright
    on your hand, you move your hand (cart) to stay under the stick (pendulum),
    effectively changing the target angle dynamically.

    The key insight: controlling angle alone causes the cart to run away.
    By adding position control, we keep the cart centered while balancing.

    Attributes:
        angle_pid: PID controller for pendulum angle (inner loop)
        position_pid: PID controller for cart position (outer loop)
        saturation_limits: Optional (min, max) force output limits
    """

    def __init__(
        self,
        angle_pid_gains: Tuple[float, float, float],
        position_pid_gains: Tuple[float, float, float],
        dt: float,
        saturation_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """Initialize the cascade PID controller.

        Args:
            angle_pid_gains: (Kp, Ki, Kd) for inner angle control loop
            position_pid_gains: (Kp, Ki, Kd) for outer position control loop
            dt: Control timestep (s)
            saturation_limits: Optional (min, max) output limits for force

        Design Guidelines:
            - Inner loop (angle) should be faster/more aggressive
            - Outer loop (position) should be slower/gentler
            - Typical ratio: inner loop 5-10x faster than outer loop
        """
        # Inner loop: controls angle (fast dynamics)
        self.angle_pid = PIDController(
            Kp=angle_pid_gains[0],
            Ki=angle_pid_gains[1],
            Kd=angle_pid_gains[2],
            dt=dt,
            saturation_limits=None  # No saturation on intermediate signal
        )

        # Outer loop: controls position by setting angle target (slow dynamics)
        self.position_pid = PIDController(
            Kp=position_pid_gains[0],
            Ki=position_pid_gains[1],
            Kd=position_pid_gains[2],
            dt=dt,
            saturation_limits=(-0.15, 0.15)  # Limit angle setpoint to ±8.6 degrees
        )

        self.saturation_limits = saturation_limits

    def compute(self, state: np.ndarray, position_setpoint: float = 0.0) -> float:
        """Compute cascade PID control signal.

        The control flow:
        1. Outer loop: position error → desired angle offset
        2. Inner loop: (desired angle - actual angle) → force

        Args:
            state: Current state vector [x, x_dot, theta, theta_dot]
            position_setpoint: Desired cart position (default: 0.0 m)

        Returns:
            Control force (N), saturated if limits are set
        """
        x, x_dot, theta, theta_dot = state

        # Outer loop: position error determines angle setpoint
        # Physics: When pendulum leans left (θ < 0), cart accelerates right
        #          When pendulum leans right (θ > 0), cart accelerates left
        # Strategy: If cart is too far left (x < 0), lean left (θ < 0) to push it right
        #           If cart is too far right (x > 0), lean right (θ > 0) to push it left
        # This is counterintuitive but correct for inverted pendulum!
        angle_setpoint = -self.position_pid.compute(position_setpoint, x)

        # Inner loop: angle error determines force
        # The angle_setpoint is the target angle needed to bring cart back
        force = self.angle_pid.compute(angle_setpoint, theta)

        # Apply force saturation
        if self.saturation_limits is not None:
            min_limit, max_limit = self.saturation_limits
            force = np.clip(force, min_limit, max_limit)

        return force

    def reset(self) -> None:
        """Reset both inner and outer loop controllers.

        Clears integral accumulators and previous errors in both PIDs.
        """
        self.angle_pid.reset()
        self.position_pid.reset()

    def get_state(self) -> dict:
        """Get the internal state of both controllers.

        Returns:
            Dictionary containing:
                - 'angle_pid_state': Inner loop PID state
                - 'position_pid_state': Outer loop PID state
        """
        return {
            'angle_pid_state': self.angle_pid.get_state(),
            'position_pid_state': self.position_pid.get_state()
        }

    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"CascadePIDController(\n"
            f"  angle_pid: Kp={self.angle_pid.Kp}, Ki={self.angle_pid.Ki}, "
            f"Kd={self.angle_pid.Kd}\n"
            f"  position_pid: Kp={self.position_pid.Kp}, Ki={self.position_pid.Ki}, "
            f"Kd={self.position_pid.Kd}\n"
            f"  saturation_limits={self.saturation_limits})"
        )


class GainScheduledCascadePID:
    """Gain-scheduled cascade PID controller for inverted pendulum.

    This controller adapts its gains based on the system state, using different
    control strategies for different operating regions:

    - Large angle (emergency): Aggressive angle control to prevent falling
    - Medium angle (stabilizing): Balanced control for convergence
    - Small angle (fine-tuning): Gentle control with position correction

    Gain scheduling improves performance across the entire operating range,
    addressing PID's limitation of fixed gains that can't adapt to varying dynamics.

    Attributes:
        gain_sets: Dictionary of gain sets for different operating regions
        saturation_limits: Force output limits
        current_mode: Current operating mode for diagnostics
    """

    def __init__(
        self,
        dt: float,
        saturation_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """Initialize gain-scheduled cascade PID controller.

        Args:
            dt: Control timestep (s)
            saturation_limits: Optional (min, max) output limits for force
        """
        self.dt = dt
        self.saturation_limits = saturation_limits
        self.current_mode = "INIT"

        # Define gain sets for different operating regions
        # Format: (angle_Kp, angle_Ki, angle_Kd, position_Kp, position_Ki, position_Kd)
        self.gain_sets = {
            # Emergency mode: |theta| > 15° - Focus on angle, ignore position
            'emergency': {
                'angle': (150.0, 0.5, 40.0),     # Very aggressive angle control
                'position': (0.0, 0.0, 0.0),     # No position control (would interfere)
                'angle_limit': 0.5               # Wider angle setpoint limit
            },
            # Stabilization mode: 5° < |theta| < 15° - Balance angle and position
            'stabilizing': {
                'angle': (120.0, 2.0, 35.0),     # Strong angle control
                'position': (0.8, 0.0, 3.0),     # Moderate position control
                'angle_limit': 0.2               # ~11° limit
            },
            # Fine-tuning mode: |theta| < 5° - Refine position, gentle angle
            'fine_tuning': {
                'angle': (80.0, 5.0, 25.0),      # Moderate angle control
                'position': (1.5, 0.1, 5.0),     # Strong position control
                'angle_limit': 0.15              # ~8.6° limit
            }
        }

        # Create PID controllers (will be updated with gain scheduling)
        self._create_controllers('fine_tuning')

    def _create_controllers(self, mode: str) -> None:
        """Create or update PID controllers for specified mode.

        Args:
            mode: Operating mode ('emergency', 'stabilizing', or 'fine_tuning')
        """
        gains = self.gain_sets[mode]

        # Inner loop: angle control
        self.angle_pid = PIDController(
            Kp=gains['angle'][0],
            Ki=gains['angle'][1],
            Kd=gains['angle'][2],
            dt=self.dt,
            saturation_limits=None
        )

        # Outer loop: position control
        self.position_pid = PIDController(
            Kp=gains['position'][0],
            Ki=gains['position'][1],
            Kd=gains['position'][2],
            dt=self.dt,
            saturation_limits=(-gains['angle_limit'], gains['angle_limit'])
        )

        self.current_mode = mode

    def _select_mode(self, theta: float) -> str:
        """Select operating mode based on pendulum angle.

        Args:
            theta: Pendulum angle (rad)

        Returns:
            Operating mode string
        """
        theta_abs = abs(theta)

        if theta_abs > np.deg2rad(15):  # > 15 degrees
            return 'emergency'
        elif theta_abs > np.deg2rad(5):  # 5-15 degrees
            return 'stabilizing'
        else:  # < 5 degrees
            return 'fine_tuning'

    def compute(self, state: np.ndarray, position_setpoint: float = 0.0) -> float:
        """Compute gain-scheduled cascade PID control signal.

        Automatically selects appropriate gains based on current state and
        applies cascade control strategy.

        Args:
            state: Current state vector [x, x_dot, theta, theta_dot]
            position_setpoint: Desired cart position (default: 0.0 m)

        Returns:
            Control force (N), saturated if limits are set
        """
        x, x_dot, theta, theta_dot = state

        # Select and apply appropriate gain set
        desired_mode = self._select_mode(theta)
        if desired_mode != self.current_mode:
            # Preserve controller states when switching modes
            old_angle_state = self.angle_pid.get_state()
            old_position_state = self.position_pid.get_state()

            self._create_controllers(desired_mode)

            # Restore states to prevent discontinuities
            # (Only restore integral to maintain smoothness)
            self.angle_pid._integral = old_angle_state['integral'] * 0.5  # Decay integral
            self.position_pid._integral = old_position_state['integral'] * 0.5

        # Cascade control with corrected sign
        angle_setpoint = -self.position_pid.compute(position_setpoint, x)
        force = self.angle_pid.compute(angle_setpoint, theta)

        # Apply force saturation
        if self.saturation_limits is not None:
            force = np.clip(force, *self.saturation_limits)

        return force

    def reset(self) -> None:
        """Reset controller to initial state."""
        self._create_controllers('fine_tuning')

    def get_current_mode(self) -> str:
        """Get the current operating mode.

        Returns:
            Current mode string
        """
        return self.current_mode

    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"GainScheduledCascadePID(\n"
            f"  current_mode={self.current_mode}\n"
            f"  angle_pid: Kp={self.angle_pid.Kp}, Ki={self.angle_pid.Ki}, "
            f"Kd={self.angle_pid.Kd}\n"
            f"  position_pid: Kp={self.position_pid.Kp}, Ki={self.position_pid.Ki}, "
            f"Kd={self.position_pid.Kd})"
        )
