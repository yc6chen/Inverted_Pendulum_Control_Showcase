"""
Filters Module for Inverted Pendulum System

This module implements state estimation algorithms for the cart-pendulum system.
The Kalman Filter provides optimal state estimation from noisy sensor measurements,
enabling robust control even with imperfect observations.

The filter combines:
- Prediction step: Uses system dynamics to predict the next state
- Update step: Corrects prediction using actual measurements
"""

import numpy as np
from typing import Optional


class KalmanFilter:
    """Discrete-time Kalman Filter for optimal state estimation.

    The Kalman Filter is an optimal recursive estimator that combines:
    1. A dynamic model of the system (prediction)
    2. Noisy measurements (correction)

    It operates in two steps:
    - Predict: x_k|k-1 = A*x_k-1 + B*u_k-1
               P_k|k-1 = A*P_k-1*A^T + Q

    - Update:  K_k = P_k|k-1*C^T*(C*P_k|k-1*C^T + R)^-1
               x_k = x_k|k-1 + K_k*(y_k - C*x_k|k-1)
               P_k = (I - K_k*C)*P_k|k-1

    where:
        x: State estimate
        P: Estimate covariance matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        K: Kalman gain

    Attributes:
        A_d: Discretized state transition matrix (n x n)
        B_d: Discretized control input matrix (n x m)
        C_d: Measurement matrix (p x n)
        Q: Process noise covariance matrix (n x n)
        R: Measurement noise covariance matrix (p x p)
        state_estimate: Current state estimate (n,)
        estimate_covariance: Current estimate covariance matrix (n x n)
    """

    def __init__(
        self,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray
    ) -> None:
        """Initialize the Kalman Filter.

        Args:
            A_d: Discretized state transition matrix (n x n)
            B_d: Discretized control input matrix (n x m)
            C_d: Measurement matrix (p x n) - maps state to measurements
            Q: Process noise covariance matrix (n x n, positive semi-definite)
            R: Measurement noise covariance matrix (p x p, positive definite)
            x0: Initial state estimate (n,)

        Raises:
            ValueError: If matrix dimensions are incompatible or invalid
        """
        # Validate matrix shapes and compatibility
        self._validate_matrix_shapes(A_d, B_d, C_d, Q, R, x0)

        # Store system matrices
        self.A_d = A_d.copy()
        self.B_d = B_d.copy()
        self.C_d = C_d.copy()
        self.Q = Q.copy()
        self.R = R.copy()

        # Initialize state estimate
        self.state_estimate = x0.copy()

        # Initialize estimate covariance
        # Start with moderate uncertainty
        n_states = x0.shape[0]
        self.estimate_covariance = np.eye(n_states) * 0.1

    def _validate_matrix_shapes(
        self,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray
    ) -> None:
        """Validate that all matrices have compatible dimensions.

        Args:
            A_d: State transition matrix
            B_d: Control input matrix
            C_d: Measurement matrix
            Q: Process noise covariance
            R: Measurement noise covariance
            x0: Initial state

        Raises:
            ValueError: If any matrix dimensions are incompatible
        """
        # Get dimensions
        if x0.ndim != 1:
            raise ValueError(
                f"Initial state x0 must be 1-dimensional. Got shape {x0.shape}"
            )

        n_states = x0.shape[0]

        # Validate A_d is square with correct size
        if A_d.shape != (n_states, n_states):
            raise ValueError(
                f"A_d must be {n_states}x{n_states}. Got shape {A_d.shape}"
            )

        # Validate B_d has correct number of rows
        if B_d.shape[0] != n_states:
            raise ValueError(
                f"B_d must have {n_states} rows. Got shape {B_d.shape}"
            )

        # Get control input dimension
        n_inputs = B_d.shape[1]

        # Validate C_d has correct number of columns
        if C_d.shape[1] != n_states:
            raise ValueError(
                f"C_d must have {n_states} columns. Got shape {C_d.shape}"
            )

        # Get measurement dimension
        n_measurements = C_d.shape[0]

        # Validate Q is square with correct size
        if Q.shape != (n_states, n_states):
            raise ValueError(
                f"Process noise Q must be {n_states}x{n_states}. Got shape {Q.shape}"
            )

        # Validate Q is symmetric
        if not np.allclose(Q, Q.T):
            raise ValueError("Process noise Q must be symmetric")

        # Validate Q is positive semi-definite
        try:
            q_eigenvalues = np.linalg.eigvals(Q)
            if np.any(q_eigenvalues < -1e-10):
                raise ValueError(
                    f"Process noise Q must be positive semi-definite. "
                    f"Got eigenvalues: {q_eigenvalues}"
                )
        except np.linalg.LinAlgError:
            raise ValueError("Failed to compute eigenvalues of Q")

        # Validate R is square with correct size
        if R.shape != (n_measurements, n_measurements):
            raise ValueError(
                f"Measurement noise R must be {n_measurements}x{n_measurements}. "
                f"Got shape {R.shape}"
            )

        # Validate R is symmetric
        if not np.allclose(R, R.T):
            raise ValueError("Measurement noise R must be symmetric")

        # Validate R is positive definite
        try:
            r_eigenvalues = np.linalg.eigvals(R)
            if np.any(r_eigenvalues <= 1e-10):
                raise ValueError(
                    f"Measurement noise R must be positive definite. "
                    f"Got eigenvalues: {r_eigenvalues}"
                )
        except np.linalg.LinAlgError:
            raise ValueError("Failed to compute eigenvalues of R")

    def predict(self, u: np.ndarray) -> None:
        """Perform the prediction step of the Kalman Filter.

        Uses the system dynamics to predict the next state and covariance:
            x_k|k-1 = A*x_k-1 + B*u_k-1
            P_k|k-1 = A*P_k-1*A^T + Q

        Args:
            u: Control input vector (m,)

        Raises:
            ValueError: If control input dimension is incorrect
        """
        # Validate control input dimension
        expected_input_dim = self.B_d.shape[1]
        if u.ndim == 0:
            # Scalar input - convert to 1D array
            u = np.array([u])
        elif u.ndim != 1:
            raise ValueError(
                f"Control input u must be 1-dimensional. Got shape {u.shape}"
            )

        if u.shape[0] != expected_input_dim:
            raise ValueError(
                f"Control input u must have {expected_input_dim} elements. "
                f"Got {u.shape[0]}"
            )

        # Predict state
        self.state_estimate = self.A_d @ self.state_estimate + self.B_d @ u

        # Predict covariance
        self.estimate_covariance = (
            self.A_d @ self.estimate_covariance @ self.A_d.T + self.Q
        )

    def update(self, y: np.ndarray) -> None:
        """Perform the update step of the Kalman Filter.

        Corrects the prediction using the measurement:
            K_k = P_k|k-1*C^T*(C*P_k|k-1*C^T + R)^-1
            x_k = x_k|k-1 + K_k*(y_k - C*x_k|k-1)
            P_k = (I - K_k*C)*P_k|k-1

        Args:
            y: Measurement vector (p,)

        Raises:
            ValueError: If measurement dimension is incorrect
        """
        # Validate measurement dimension
        expected_meas_dim = self.C_d.shape[0]
        if y.ndim != 1:
            raise ValueError(
                f"Measurement y must be 1-dimensional. Got shape {y.shape}"
            )

        if y.shape[0] != expected_meas_dim:
            raise ValueError(
                f"Measurement y must have {expected_meas_dim} elements. "
                f"Got {y.shape[0]}"
            )

        # Compute innovation covariance: S = C*P*C^T + R
        S = self.C_d @ self.estimate_covariance @ self.C_d.T + self.R

        # Compute Kalman gain: K = P*C^T*S^-1
        # Use robust inversion with fallback to pseudo-inverse
        try:
            K = self.estimate_covariance @ self.C_d.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudo-inverse for numerical stability
            K = self.estimate_covariance @ self.C_d.T @ np.linalg.pinv(S)

        # Compute innovation: y - C*x
        innovation = y - self.C_d @ self.state_estimate

        # Update state estimate
        self.state_estimate += K @ innovation

        # Update estimate covariance: P = (I - K*C)*P
        # Use Joseph form for numerical stability
        I = np.eye(self.estimate_covariance.shape[0])
        I_KC = I - K @ self.C_d
        self.estimate_covariance = I_KC @ self.estimate_covariance

    def step(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform one complete prediction-update cycle.

        Convenience method that combines predict() and update() in sequence.

        Args:
            u: Control input vector (m,)
            y: Measurement vector (p,)

        Returns:
            Updated state estimate (n,)
        """
        self.predict(u)
        self.update(y)
        return self.state_estimate.copy()

    def get_state_estimate(self) -> np.ndarray:
        """Get the current state estimate.

        Returns:
            Current state estimate vector (n,)
        """
        return self.state_estimate.copy()

    def get_estimate_covariance(self) -> np.ndarray:
        """Get the current estimate covariance matrix.

        The diagonal elements represent the variance (uncertainty) in each
        state estimate. Lower values indicate higher confidence.

        Returns:
            Current estimate covariance matrix (n x n)
        """
        return self.estimate_covariance.copy()

    def reset(self, x0: np.ndarray, P0: Optional[np.ndarray] = None) -> None:
        """Reset the filter to a new initial state.

        Args:
            x0: New initial state estimate (n,)
            P0: New initial covariance matrix (n x n). If None, uses default.

        Raises:
            ValueError: If dimensions are incompatible
        """
        if x0.shape[0] != self.state_estimate.shape[0]:
            raise ValueError(
                f"Reset state x0 must have {self.state_estimate.shape[0]} elements. "
                f"Got {x0.shape[0]}"
            )

        self.state_estimate = x0.copy()

        if P0 is not None:
            expected_shape = (x0.shape[0], x0.shape[0])
            if P0.shape != expected_shape:
                raise ValueError(
                    f"Reset covariance P0 must have shape {expected_shape}. "
                    f"Got {P0.shape}"
                )
            self.estimate_covariance = P0.copy()
        else:
            # Reset to default initial covariance
            self.estimate_covariance = np.eye(x0.shape[0]) * 0.1

    def __repr__(self) -> str:
        """String representation of the filter."""
        n_states = self.state_estimate.shape[0]
        n_measurements = self.C_d.shape[0]
        return (
            f"KalmanFilter(n_states={n_states}, n_measurements={n_measurements}, "
            f"current_estimate={self.state_estimate})"
        )
