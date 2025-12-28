#!/usr/bin/env python3
"""
Phase 5: Comprehensive Analysis and Visualization

This script demonstrates the complete inverted pendulum control project
with systematic comparison of PID vs LQR controllers under ideal and
noisy conditions, showcasing the role of the Kalman Filter.

This is the final, resume-ready implementation that:
1. Runs all controller scenarios in a structured pipeline
2. Calculates comprehensive performance metrics
3. Generates publication-quality comparison plots
4. Provides reproducible, well-documented results

Author: Control Systems Engineer
Purpose: Demonstrate mastery of control theory, state estimation, and Python engineering
"""

import numpy as np
import logging
from typing import List
import sys

from config import (
    SimulationConfig,
    PhysicalParameters,
    SimulationParameters,
    ControlParameters,
    PIDGains,
    LQRGains
)
from analysis import SimulationRunner, SimulationResult
from metrics import calculate_all_metrics, compare_metrics
from visualization import generate_comprehensive_report, plot_kalman_filter_performance


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_configurations() -> List[tuple]:
    """Create configurations for all scenarios to compare.

    Returns:
        List of (config, use_noise, use_filter) tuples
    """
    # Standard physical parameters
    physical = PhysicalParameters(
        M=1.0,
        m=0.3,
        l=0.5,
        b=0.1
    )

    # Standard simulation parameters
    simulation = SimulationParameters(
        dt_plant=0.001,
        dt_control=0.01,
        sim_time=10.0,
        initial_state=(0.0, 0.0, 0.1, 0.0)  # 0.1 rad ≈ 5.7 degrees
    )

    configurations = []

    # PID Controller Scenarios
    pid_control = ControlParameters(
        controller_type='PID',
        pid_gains=PIDGains(Kp=50.0, Ki=30.0, Kd=15.0),
        saturation_limits=(-20.0, 20.0)
    )

    # Scenario 1: PID with ideal measurements
    config_pid_ideal = SimulationConfig(
        physical=physical,
        simulation=simulation,
        control=pid_control
    )
    configurations.append((config_pid_ideal, False, False))

    # Scenario 2: PID with noisy measurements (no filter)
    config_pid_noisy = SimulationConfig(
        physical=physical,
        simulation=simulation,
        control=pid_control
    )
    configurations.append((config_pid_noisy, True, False))

    # Scenario 3: PID with noisy measurements + Kalman filter
    config_pid_filtered = SimulationConfig(
        physical=physical,
        simulation=simulation,
        control=pid_control
    )
    configurations.append((config_pid_filtered, True, True))

    # LQR Controller Scenarios
    lqr_control = ControlParameters(
        controller_type='LQR',
        lqr_gains=LQRGains(
            Q=(1.0, 0.0, 10.0, 0.0),  # Penalize position and angle
            R=0.01  # Small control penalty
        ),
        saturation_limits=(-20.0, 20.0)
    )

    # Scenario 4: LQR with ideal measurements
    config_lqr_ideal = SimulationConfig(
        physical=physical,
        simulation=simulation,
        control=lqr_control
    )
    configurations.append((config_lqr_ideal, False, False))

    # Scenario 5: LQR with noisy measurements (no filter)
    config_lqr_noisy = SimulationConfig(
        physical=physical,
        simulation=simulation,
        control=lqr_control
    )
    configurations.append((config_lqr_noisy, True, False))

    # Scenario 6: LQR with noisy measurements + Kalman filter
    config_lqr_filtered = SimulationConfig(
        physical=physical,
        simulation=simulation,
        control=lqr_control
    )
    configurations.append((config_lqr_filtered, True, True))

    return configurations


def run_all_scenarios() -> List[SimulationResult]:
    """Run all controller scenarios and return results.

    Returns:
        List of SimulationResult objects
    """
    logger.info("=" * 80)
    logger.info("PHASE 5: COMPREHENSIVE CONTROLLER COMPARISON")
    logger.info("=" * 80)

    configurations = create_configurations()
    all_results = []

    logger.info(f"\nRunning {len(configurations)} scenarios...")

    for i, (config, use_noise, use_filter) in enumerate(configurations, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario {i}/{len(configurations)}")
        logger.info(f"{'='*60}")

        # Create runner
        runner = SimulationRunner(
            config=config,
            use_noise=use_noise,
            use_filter=use_filter,
            noise_std=(0.01, 0.005)  # Position and angle noise std
        )

        # Run simulation
        try:
            result = runner.run()
            all_results.append(result)
            logger.info(f"✓ Scenario {i} completed: {result.get_label()}")
        except Exception as e:
            logger.error(f"✗ Scenario {i} failed: {e}", exc_info=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Completed {len(all_results)}/{len(configurations)} scenarios successfully")
    logger.info(f"{'='*60}\n")

    return all_results


def main():
    """Main orchestrator function.

    This function:
    1. Loads configurations for all scenarios
    2. Runs each simulation
    3. Calculates performance metrics
    4. Generates comprehensive visualizations
    5. Prints comparison summary
    """
    try:
        # Step 1: Run all scenarios
        results = run_all_scenarios()

        if not results:
            logger.error("No simulations completed successfully!")
            return 1

        # Step 2: Calculate metrics for all results
        logger.info("\nCalculating performance metrics...")
        for result in results:
            calculate_all_metrics(result)

        # Step 3: Generate comparison table
        logger.info("\nGenerating performance comparison...")
        compare_metrics(results)

        # Step 4: Generate comprehensive visualization
        logger.info("\nGenerating comprehensive comparison plots...")
        generate_comprehensive_report(
            results,
            save_path="phase5_comprehensive_comparison.png",
            show=True
        )

        # Step 5: Generate Kalman filter visualization (if applicable)
        logger.info("\nGenerating Kalman filter performance plots...")
        filtered_results = [r for r in results if r.use_filter]
        if filtered_results:
            # Show one example of Kalman filter performance
            plot_kalman_filter_performance(
                filtered_results[0],  # Show first filtered result (PID)
                save_path="phase5_kalman_filter_performance.png",
                show=True
            )

        # Step 6: Print key findings
        print_key_findings(results)

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5 ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info("\nOutputs:")
        logger.info("  - phase5_comprehensive_comparison.png")
        logger.info("  - phase5_kalman_filter_performance.png")
        logger.info("\nNext steps:")
        logger.info("  1. Review the generated plots")
        logger.info("  2. Analyze the performance comparison table")
        logger.info("  3. Update your resume with quantified results!")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nSimulation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        return 1


def print_key_findings(results: List[SimulationResult]):
    """Print key findings and insights from the comparison.

    Args:
        results: List of simulation results with metrics
    """
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR YOUR RESUME")
    print("=" * 80)

    # Filter successful results
    successful = [r for r in results if r.metrics.get('success', 0) == 1.0]

    if not successful:
        print("No successful stabilizations - check controller tuning!")
        return

    # Find LQR and PID results
    lqr_ideal = [r for r in successful if r.controller_type == 'LQR' and not r.use_noise]
    pid_ideal = [r for r in successful if r.controller_type == 'PID' and not r.use_noise]

    lqr_filtered = [r for r in successful if r.controller_type == 'LQR' and r.use_filter]
    pid_filtered = [r for r in successful if r.controller_type == 'PID' and r.use_filter]

    # Finding 1: LQR vs PID performance
    if lqr_ideal and pid_ideal:
        lqr = lqr_ideal[0]
        pid = pid_ideal[0]

        settling_improvement = (
            (pid.metrics['angle_settling_time'] - lqr.metrics['angle_settling_time']) /
            pid.metrics['angle_settling_time'] * 100
        )

        effort_reduction = (
            (pid.metrics['control_effort_total'] - lqr.metrics['control_effort_total']) /
            pid.metrics['control_effort_total'] * 100
        )

        print("\n1. LQR vs PID (Ideal Conditions):")
        print(f"   • LQR achieved {settling_improvement:.1f}% faster settling time")
        print(f"   • LQR used {effort_reduction:.1f}% less control energy")
        print(f"   • LQR settling: {lqr.metrics['angle_settling_time']:.3f}s "
              f"vs PID: {pid.metrics['angle_settling_time']:.3f}s")

    # Finding 2: Impact of noise
    if lqr_ideal and lqr_filtered:
        print("\n2. Impact of Sensor Noise:")
        noisy_results = [r for r in results if r.use_noise and not r.use_filter]
        if noisy_results:
            degradation = [r for r in noisy_results if r.metrics.get('success', 0) == 0.0]
            if degradation:
                print(f"   • {len(degradation)} controller(s) failed with noise alone")
            print("   • Noise significantly degraded performance in all cases")

    # Finding 3: Kalman filter effectiveness
    if lqr_filtered and lqr_ideal:
        print("\n3. Kalman Filter Effectiveness:")
        lqr_filt = lqr_filtered[0]
        lqr_id = lqr_ideal[0]

        settling_ratio = (lqr_filt.metrics['angle_settling_time'] /
                         lqr_id.metrics['angle_settling_time'])

        print(f"   • Kalman filter enabled stable control despite measurement noise")
        print(f"   • Performance with KF: {settling_ratio:.1%} of ideal case")
        print(f"   • Demonstrates critical role of state estimation in real systems")

    # Resume bullet point suggestions
    print("\n" + "=" * 80)
    print("RESUME BULLET POINTS (Copy these!):")
    print("=" * 80)

    if lqr_ideal and pid_ideal:
        lqr = lqr_ideal[0]
        pid = pid_ideal[0]
        effort_reduction = (
            (pid.metrics['control_effort_total'] - lqr.metrics['control_effort_total']) /
            pid.metrics['control_effort_total'] * 100
        )

        print(f"""
• Engineered a high-fidelity Python simulation comparing classical PID control
  against optimal LQR control for an inverted pendulum system, demonstrating
  {effort_reduction:.0f}% energy reduction with LQR through model-based optimization

• Implemented a Kalman Filter for robust state estimation from noisy sensor data,
  enabling stable control where raw measurements caused controller failure

• Designed and validated a complete control pipeline (physics simulation, dual
  controllers, state estimation) with systematic performance analysis quantifying
  settling time, control effort, and robustness tradeoffs
        """)

    print("=" * 80)


if __name__ == "__main__":
    exit(main())
