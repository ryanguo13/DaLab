# Pneumatic System State Space Analysis Tool

This project provides a set of tools for state space analysis of pneumatic systems, including linearization analysis, state feedback controller design, nonlinear system simulation, and PID control simulation.

## Project Structure

```
ss/
├── main.py               # Main program entry
├── modules/              # Modules directory
│   ├── __init__.py       # Package initialization
│   ├── pid_controller.py # PID controller implementation
│   ├── state_space_model.py # State space model
│   ├── system_dynamics.py # System dynamics
│   └── visualization.py  # Visualization tools
└── results/              # Output directory (auto-created)
```

## Features

- **Linear System Analysis**: Linearize system model, analyze stability, controllability and observability
- **State Feedback Controller Design**: Design state feedback controllers using pole placement
- **Nonlinear System Simulation**: Simulate full nonlinear system dynamics
- **Linear vs Nonlinear Comparison**: Compare responses between linearized and nonlinear models
- **PID Control Simulation**: Position control simulation using fuzzy PID controller
- **Visualization Tools**: Generate system responses, phase space trajectories, root locus and Bode plots

## Dependencies

Ensure these Python libraries are installed:

```bash
pip install numpy matplotlib scipy control
```

## Usage

### Basic Usage

```bash
python main.py
```

By default, the program runs all analysis modes.

### Specifying Analysis Mode

```bash
# Run only linear analysis
python main.py --mode linear

# Run only nonlinear analysis
python main.py --mode nonlinear

# Run linear vs nonlinear comparison
python main.py --mode compare

# Run PID control simulation
python main.py --mode pid
```

### Custom System Parameters

```bash
python main.py --m_ball 0.05 --A 0.001 --f_v 0.1 --V_tube 0.01 --L_tube 0.5 --P_s 500000
```

### Custom Controller Parameters

```bash
python main.py --desired_poles "-3,-3.5,-4,-4.5" --kp 0.3 --ki 0.0 --kd 20.0
```

### Modifying Simulation Parameters

```bash
python main.py --t_span 2.0 --dt 0.01 --x0 "0.26,0,200000,200000" --pid_duration 28.0
```

### Saving and Displaying Plots

```bash
# Save plots and generate report
python main.py --save_figs --output_dir results

# Display plots
python main.py --show_plots
```

## Custom Experiments

### Comparing Different Friction Coefficients

```bash
python main.py --mode compare --f_v 0.05 --save_figs --output_dir "results/low_friction"
python main.py --mode compare --f_v 0.2 --save_figs --output_dir "results/high_friction"
```

### Comparing Different PID Parameters

```bash
python main.py --mode pid --kp 0.1 --kd 10.0 --save_figs --output_dir "results/pid_low"
python main.py --mode pid --kp 0.5 --kd 30.0 --save_figs --output_dir "results/pid_high"
```

### Adjusting Simulation Time and Step Size

```bash
# Long duration simulation
python main.py --mode pid --pid_duration 84.0 --save_figs

# High precision simulation
python main.py --mode compare --dt 0.001 --save_figs
```

## System Parameters Description

- `m_ball`: Ball mass (kg)
- `A`: Pipe cross-sectional area (m²)
- `f_v`: Viscous friction coefficient
- `V_tube`: Total pipe volume (m³)
- `L_tube`: Pipe length (m)
- `P_s`: Supply pressure (Pa)
- `P_atm`: Atmospheric pressure (Pa)
- `x_eq`: Equilibrium position (m)
- `P_l_eq`: Left chamber equilibrium pressure (Pa)
- `P_r_eq`: Right chamber equilibrium pressure (Pa)

## Controller Parameters Description

- `desired_poles`: Desired poles (comma separated)
- `kp`: PID controller proportional gain
- `ki`: PID controller integral gain
- `kd`: PID controller derivative gain

## Generated Plots

- Linear/nonlinear model responses
- State space trajectories
- Root locus
- Bode plots
- Linear vs nonlinear model comparison
- PID control responses
