"""
Controllers Module for Inverted Pendulum System

This module implements various control strategies for stabilizing the inverted
pendulum system. Currently includes a PID controller with anti-windup logic.

The PID controller uses a discrete-time implementation suitable for real-time
control applications.
"""

import numpy as np
from typing import Optional, Tuple


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
