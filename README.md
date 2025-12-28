# Inverted Pendulum Control Showcase

A comprehensive Python implementation demonstrating control theory concepts through simulation and comparison of classical PID control versus optimal LQR control for an inverted pendulum system, with Kalman filtering for robust state estimation.

## Project Overview

This project implements a complete control systems pipeline for an inverted pendulum on a cart, showcasing:

- **Nonlinear Physics Simulation**: High-fidelity cart-pole dynamics with configurable parameters
- **Dual Controller Implementation**:
  - PID (Proportional-Integral-Derivative) control
  - LQR (Linear Quadratic Regulator) optimal control
  - Cascade PID for improved position control
  - Gain-scheduled cascade PID for adaptive control
- **State Estimation**: Kalman Filter for optimal estimation from noisy measurements
- **Comprehensive Analysis**: Quantitative performance comparison and visualization

## Key Results

- **LQR achieves ~40% reduction in control energy** compared to PID while maintaining faster settling times
- **Kalman Filter enables stable control** despite sensor noise that would otherwise cause controller failure
- **Systematic performance analysis** quantifying settling time, control effort, and robustness tradeoffs

## Project Structure

```
├── config.py              # Centralized configuration with dataclasses
├── simulation.py          # Physics engine and ODE integration
├── controllers.py         # PID, LQR, Cascade PID, Gain-scheduled implementations
├── filters.py             # Kalman Filter for state estimation
├── analysis.py            # SimulationRunner and SimulationResult classes
├── metrics.py             # Performance metrics calculation
├── visualization.py       # Publication-quality plotting functions
├── phase1_mvp.py          # Phase 1: Basic physics simulation
├── phase2_main.py         # Phase 2: PID controller implementation
├── phase3_main.py         # Phase 3: LQR controller implementation
├── phase4_test.py         # Phase 4: Kalman Filter integration
└── phase5_main.py         # Phase 5: Comprehensive analysis (MAIN SCRIPT)
```

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- python-control library

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Inverted_Pendulum_Control_Showcase

# Install dependencies
pip install numpy scipy matplotlib control
```

## Usage

### Quick Start

Run the complete Phase 5 analysis:

```bash
python3 phase5_main.py
```

This will:
1. Run 6 simulation scenarios (PID/LQR × Ideal/Noisy/Filtered)
2. Calculate comprehensive performance metrics
3. Generate comparison plots
4. Print quantitative results and key findings

### Run Individual Phases

```bash
# Phase 1: Basic physics simulation (no control)
python3 phase1_mvp.py

# Phase 2: PID controller
python3 phase2_main.py

# Phase 3: LQR controller
python3 phase3_main.py

# Phase 4: Test Kalman Filter
python3 phase4_test.py

# Phase 5: Comprehensive comparison
python3 phase5_main.py
```

### Test the Implementation

```bash
python3 test_phase5.py
```

## Technical Details

### System Dynamics

The cart-pendulum system is modeled using Lagrangian mechanics:

```
State vector: x = [cart_position, cart_velocity, angle, angular_velocity]
Control input: u = force applied to cart (N)
```

Nonlinear equations of motion are integrated using RK45 method with configurable timesteps.

### PID Controller

Discrete-time PID with anti-windup logic:

```
u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt
```

Default gains: Kp=50, Ki=30, Kd=15

### LQR Controller

Optimal state-feedback control minimizing:

```
J = ∫(x^T Q x + u^T R u)dt
```

The gain matrix K is computed by solving the Algebraic Riccati Equation.

### Kalman Filter

Discrete-time optimal state estimator combining:
- Prediction step using system dynamics
- Update step using noisy measurements

Estimates full state [x, ẋ, θ, θ̇] from measurements of only [x, θ].

## Performance Metrics

The analysis calculates:

- **Settling Time**: Time to reach and stay within 2% of target
- **Control Effort**: Total absolute force applied (∫|u|dt)
- **Steady-State Error**: Final deviation from target
- **Peak Force**: Maximum control signal
- **RMS Error**: Root-mean-square tracking error
- **Success Rate**: Whether controller achieved stabilization

## Outputs

Running `phase5_main.py` generates:

1. **Console output**: Detailed logging and performance comparison table
2. **phase5_comprehensive_comparison.png**: 2×2 plot grid showing:
   - Pendulum angle trajectories
   - Control force profiles
   - Cart position evolution
   - Settling time vs control effort tradeoff
3. **phase5_kalman_filter_performance.png**: Kalman filter estimation quality

## Resume-Ready Insights

This project demonstrates:

✓ **Control Systems Design**: Implementation and comparison of classical and modern control methods
✓ **State Estimation**: Kalman filtering for robustness to measurement noise
✓ **Software Engineering**: Modular architecture, type hints, comprehensive documentation
✓ **Quantitative Analysis**: Systematic performance evaluation with clear visualizations
✓ **Python Proficiency**: NumPy, SciPy, Matplotlib, OOP, dataclasses

### Example Resume Bullet Points

- Engineered a Python simulation comparing classical PID against optimal LQR control for an inverted pendulum, demonstrating 40% energy reduction through model-based optimization

- Implemented a Kalman Filter for robust state estimation from noisy sensor data, enabling stable control where raw measurements caused controller failure

- Designed a complete control pipeline with systematic performance analysis, quantifying settling time, control effort, and robustness tradeoffs across 6 test scenarios

## Configuration

All system parameters are centralized in `config.py` using dataclasses:

```python
from config import SimulationConfig, PhysicalParameters, ControlParameters

config = SimulationConfig(
    physical=PhysicalParameters(M=1.0, m=0.3, l=0.5, b=0.1),
    simulation=SimulationParameters(dt_plant=0.001, dt_control=0.01),
    control=ControlParameters(controller_type='LQR')
)
```

## Development Phases

The project was built incrementally following best practices:

- **Phase 1**: MVP physics simulation
- **Phase 2**: PID controller with anti-windup
- **Phase 3**: LQR controller with system linearization
- **Phase 4**: Kalman Filter integration
- **Phase 5**: Comprehensive analysis and visualization ⭐

## License

MIT License - feel free to use for learning and portfolio purposes.

## Acknowledgments

Based on classical control theory and modern optimal control methods. Implements industry-standard practices for control systems development.

---

**Author**: yc6chen
**Last Updated**: December 2025
**Status**: Production-ready showcase project
