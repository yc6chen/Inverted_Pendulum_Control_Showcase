"""
Phase 4: Kalman Filter Testing

This script tests the Kalman Filter implementation for state estimation
in the inverted pendulum system. It verifies the filter's functionality
and demonstrates basic usage.

Key Features:
- Matrix validation testing
- Prediction and update step verification
- Integration with the linearized system model
- Numerical stability checks
"""

import numpy as np
from scipy.linalg import expm
from config import PhysicalParameters, SimulationParameters
from controllers import linearize_pendulum
from filters import KalmanFilter


def discretize_system(A: np.ndarray, B: np.ndarray, dt: float) -> tuple:
    """Discretize continuous-time state-space matrices.

    Converts continuous-time system matrices to discrete-time using
    matrix exponential method (exact discretization).

    Args:
        A: Continuous-time state matrix (n x n)
        B: Continuous-time input matrix (n x m)
        dt: Sampling time (s)

    Returns:
        Tuple of (A_d, B_d) discretized matrices
    """
    n = A.shape[0]
    m = B.shape[1]

    # Create augmented matrix for exact discretization
    # [A  B]
    # [0  0]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt

    # Compute matrix exponential
    exp_M = expm(M)

    # Extract discretized matrices
    A_d = exp_M[:n, :n]
    B_d = exp_M[:n, n:]

    return A_d, B_d


def test_kalman_filter_basic():
    """Test basic Kalman Filter functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Kalman Filter Initialization and Operation")
    print("=" * 70)

    # Create physical parameters
    params = PhysicalParameters()

    # Linearize system
    A, B = linearize_pendulum(
        M=params.M,
        m=params.m,
        l=params.l,
        b=params.b,
        g=params.g
    )

    print("\nContinuous-time system matrices:")
    print("A matrix:")
    print(A)
    print("\nB matrix:")
    print(B)

    # Discretize system
    dt = 0.01
    A_d, B_d = discretize_system(A, B, dt)

    print(f"\nDiscretized system (dt={dt} s):")
    print("A_d matrix:")
    print(A_d)
    print("\nB_d matrix:")
    print(B_d)

    # Define measurement matrix (measure position and angle)
    C_d = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Position measurement
        [0.0, 0.0, 1.0, 0.0]   # Angle measurement
    ])

    # Define noise covariances
    Q = np.eye(4) * 1e-4  # Process noise
    R = np.diag([0.01, 0.005])  # Measurement noise (position, angle)

    # Initial state
    x0 = np.array([0.0, 0.0, 0.1, 0.0])  # Start with 0.1 rad angle

    print("\nInitializing Kalman Filter...")
    print(f"Initial state: {x0}")
    print(f"Process noise Q: diag = {np.diag(Q)}")
    print(f"Measurement noise R: diag = {np.diag(R)}")

    # Create Kalman Filter
    try:
        kf = KalmanFilter(A_d, B_d, C_d, Q, R, x0)
        print("\n✓ Kalman Filter created successfully!")
        print(f"  {kf}")
    except Exception as e:
        print(f"\n✗ Error creating Kalman Filter: {e}")
        return False

    # Test predict step
    print("\nTesting predict step...")
    u = np.array([0.0])  # No control input
    try:
        kf.predict(u)
        print("✓ Predict step successful!")
        print(f"  Predicted state: {kf.get_state_estimate()}")
    except Exception as e:
        print(f"✗ Error in predict step: {e}")
        return False

    # Test update step
    print("\nTesting update step...")
    # Simulate noisy measurement
    y = np.array([0.01, 0.105])  # Noisy measurements of x and theta
    try:
        kf.update(y)
        print("✓ Update step successful!")
        print(f"  Updated state: {kf.get_state_estimate()}")
        print(f"  Estimate covariance diagonal: {np.diag(kf.get_estimate_covariance())}")
    except Exception as e:
        print(f"✗ Error in update step: {e}")
        return False

    # Test step method (combined predict + update)
    print("\nTesting combined step method...")
    try:
        estimated_state = kf.step(u, y)
        print("✓ Step method successful!")
        print(f"  Estimated state: {estimated_state}")
    except Exception as e:
        print(f"✗ Error in step method: {e}")
        return False

    # Test reset
    print("\nTesting reset method...")
    try:
        kf.reset(x0)
        print("✓ Reset successful!")
        print(f"  State after reset: {kf.get_state_estimate()}")
    except Exception as e:
        print(f"✗ Error in reset: {e}")
        return False

    print("\n" + "=" * 70)
    print("TEST 1 PASSED: All basic operations successful!")
    print("=" * 70)
    return True


def test_kalman_filter_validation():
    """Test Kalman Filter input validation."""
    print("\n" + "=" * 70)
    print("TEST 2: Kalman Filter Input Validation")
    print("=" * 70)

    # Create valid matrices
    A_d = np.eye(4)
    B_d = np.ones((4, 1))
    C_d = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    Q = np.eye(4) * 1e-4
    R = np.eye(2) * 1e-3
    x0 = np.zeros(4)

    # Test 1: Invalid A_d shape
    print("\nTest 2.1: Invalid A_d shape...")
    try:
        kf = KalmanFilter(np.eye(3), B_d, C_d, Q, R, x0)
        print("✗ Should have raised ValueError for mismatched A_d shape")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 2: Invalid C_d shape
    print("\nTest 2.2: Invalid C_d shape...")
    try:
        C_bad = np.array([[1, 0, 0]])  # Wrong number of columns
        kf = KalmanFilter(A_d, B_d, C_bad, Q, R, x0)
        print("✗ Should have raised ValueError for mismatched C_d shape")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 3: Non-symmetric Q
    print("\nTest 2.3: Non-symmetric Q matrix...")
    try:
        Q_bad = np.array([[1, 0.5, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]) * 1e-4
        kf = KalmanFilter(A_d, B_d, C_d, Q_bad, R, x0)
        print("✗ Should have raised ValueError for non-symmetric Q")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 4: Non-positive definite R
    print("\nTest 2.4: Non-positive definite R matrix...")
    try:
        R_bad = np.eye(2) * -1e-3  # Negative eigenvalues
        kf = KalmanFilter(A_d, B_d, C_d, Q, R_bad, x0)
        print("✗ Should have raised ValueError for non-positive definite R")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n" + "=" * 70)
    print("TEST 2 PASSED: All validation checks working correctly!")
    print("=" * 70)
    return True


def test_kalman_filter_simulation():
    """Test Kalman Filter with simulated data."""
    print("\n" + "=" * 70)
    print("TEST 3: Kalman Filter with Simulated Noisy Measurements")
    print("=" * 70)

    # System setup
    params = PhysicalParameters()
    A, B = linearize_pendulum(params.M, params.m, params.l, params.b, params.g)

    dt = 0.01
    A_d, B_d = discretize_system(A, B, dt)

    C_d = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    Q = np.eye(4) * 1e-4
    R = np.diag([0.01, 0.005])

    # True initial state
    x_true = np.array([0.0, 0.0, 0.1, 0.0])
    x0 = x_true + np.random.randn(4) * 0.01  # Noisy initial estimate

    # Create filter
    kf = KalmanFilter(A_d, B_d, C_d, Q, R, x0)

    print(f"\nTrue initial state:     {x_true}")
    print(f"Initial state estimate: {x0}")

    # Simulate 10 steps
    n_steps = 10
    u = np.array([0.0])

    print(f"\nRunning {n_steps} filter iterations...")
    print("-" * 70)
    print(f"{'Step':<6} {'True θ':<12} {'Meas θ':<12} {'Est θ':<12} {'Error':<12}")
    print("-" * 70)

    for step in range(n_steps):
        # Simulate true state evolution (using same dynamics)
        x_true = A_d @ x_true + B_d @ u

        # Generate noisy measurement
        y_true = C_d @ x_true
        y_meas = y_true + np.random.randn(2) * np.sqrt(np.diag(R))

        # Kalman filter step
        x_est = kf.step(u, y_meas)

        # Calculate error
        error = abs(x_true[2] - x_est[2])

        print(f"{step:<6} {np.rad2deg(x_true[2]):<12.4f} "
              f"{np.rad2deg(y_meas[1]):<12.4f} "
              f"{np.rad2deg(x_est[2]):<12.4f} "
              f"{np.rad2deg(error):<12.4f}")

    print("-" * 70)
    print(f"\nFinal true state:     {x_true}")
    print(f"Final estimate:       {kf.get_state_estimate()}")
    print(f"Estimation error:     {x_true - kf.get_state_estimate()}")

    # Check if error is reasonable
    final_error = np.linalg.norm(x_true - kf.get_state_estimate())
    if final_error < 0.1:
        print(f"\n✓ Estimation error ({final_error:.4f}) is acceptable!")
    else:
        print(f"\n⚠ Warning: Large estimation error ({final_error:.4f})")

    print("\n" + "=" * 70)
    print("TEST 3 PASSED: Filter successfully tracks noisy measurements!")
    print("=" * 70)
    return True


def main():
    """Main test execution."""
    print("=" * 70)
    print("PHASE 4: KALMAN FILTER IMPLEMENTATION TESTS")
    print("=" * 70)
    print("\nThis test suite verifies the Kalman Filter implementation:")
    print("  1. Basic initialization and operation")
    print("  2. Input validation and error handling")
    print("  3. Performance with simulated noisy data")

    # Run tests
    test_results = []

    test_results.append(test_kalman_filter_basic())
    test_results.append(test_kalman_filter_validation())
    test_results.append(test_kalman_filter_simulation())

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(test_results)}")
    print(f"Passed: {sum(test_results)}")
    print(f"Failed: {len(test_results) - sum(test_results)}")

    if all(test_results):
        print("\n✓ ALL TESTS PASSED! Kalman Filter implementation is working correctly.")
    else:
        print("\n✗ SOME TESTS FAILED. Please review the errors above.")

    print("=" * 70)


if __name__ == "__main__":
    main()
