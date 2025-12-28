#!/usr/bin/env python3
"""Quick test of Phase 5 implementation without interactive plots."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
import logging

from config import (
    SimulationConfig,
    PhysicalParameters,
    SimulationParameters,
    ControlParameters,
    PIDGains,
    LQRGains
)
from analysis import SimulationRunner
from metrics import calculate_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_simulation():
    """Test basic simulation runs without errors."""
    logger.info("Testing basic PID simulation...")

    config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(sim_time=2.0),  # Short test
        control=ControlParameters(
            controller_type='PID',
            pid_gains=PIDGains(Kp=50.0, Ki=30.0, Kd=15.0)
        )
    )

    runner = SimulationRunner(config, use_noise=False, use_filter=False)
    result = runner.run()

    logger.info(f"✓ Simulation completed: {len(result.time)} timesteps")
    logger.info(f"✓ Final angle: {result.states[-1, 2]:.4f} rad")

    return result

def test_lqr_simulation():
    """Test LQR simulation."""
    logger.info("Testing LQR simulation...")

    config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(sim_time=2.0),
        control=ControlParameters(
            controller_type='LQR',
            lqr_gains=LQRGains(Q=(1.0, 0.0, 10.0, 0.0), R=0.01)
        )
    )

    runner = SimulationRunner(config, use_noise=False, use_filter=False)
    result = runner.run()

    logger.info(f"✓ LQR simulation completed")
    return result

def test_kalman_filter():
    """Test Kalman filter integration."""
    logger.info("Testing Kalman filter...")

    config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(sim_time=2.0),
        control=ControlParameters(controller_type='LQR')
    )

    runner = SimulationRunner(config, use_noise=True, use_filter=True)
    result = runner.run()

    assert result.estimated_states is not None, "No estimated states!"
    logger.info(f"✓ Kalman filter working: {len(result.estimated_states)} estimates")

    return result

def test_metrics():
    """Test metrics calculation."""
    logger.info("Testing metrics calculation...")

    config = SimulationConfig(
        physical=PhysicalParameters(),
        simulation=SimulationParameters(sim_time=5.0)
    )

    runner = SimulationRunner(config)
    result = runner.run()

    metrics = calculate_all_metrics(result)

    assert 'angle_settling_time' in metrics
    assert 'control_effort_total' in metrics

    logger.info(f"✓ Metrics calculated: {len(metrics)} metrics")
    logger.info(f"  Settling time: {metrics['angle_settling_time']:.3f}s")
    logger.info(f"  Control effort: {metrics['control_effort_total']:.2f}")

    return metrics

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 5 IMPLEMENTATION TEST")
    print("=" * 60)

    try:
        # Test 1: Basic simulation
        result1 = test_basic_simulation()
        print()

        # Test 2: LQR
        result2 = test_lqr_simulation()
        print()

        # Test 3: Kalman filter
        result3 = test_kalman_filter()
        print()

        # Test 4: Metrics
        metrics = test_metrics()
        print()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 5 implementation is working correctly.")
        print("Run 'python3 phase5_main.py' for full analysis.")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
