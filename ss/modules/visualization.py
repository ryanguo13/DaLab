#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module
=========
For data visualization, plotting system responses, phase space trajectories and 
frequency domain analysis charts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import control as ctrl

def plot_response(response, title_prefix="System"):
    """
    Plot time domain response charts
    
    Parameters:
    response: Dictionary containing time, states, output and input
    title_prefix: Chart title prefix
    
    Returns:
    matplotlib figure object
    """
    # Extract data
    time = response['time']
    states = response['states']
    output = response['output']
    control_input = response['input']
    
    # Check if setpoint is provided in the response
    has_setpoint = 'setpoint' in response
    setpoint = response.get('setpoint', None)
    
    # Fix dimensions
    if output.ndim > 1:
        output = output.flatten()
        
    # Fix state matrix dimensions
    if states.ndim == 1:
        # If state is one-dimensional, reshape to 2D
        states = states.reshape(1, -1)
    
    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{title_prefix} Response", fontsize=16)
    
    # Position plot (output)
    axs[0, 0].plot(time, output, 'b-', linewidth=2, label='Actual Position')

    # Removed setpoint plotting as it's not used in linear analysis anymore
    # if has_setpoint:
    #     axs[0, 0].plot(time, setpoint, 'r--', linewidth=2, label='Setpoint')
    #     axs[0, 0].legend()

    axs[0, 0].set_title('Position Response')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Position (m)')
    axs[0, 0].grid(True)

    # Velocity plot (state 1)
    axs[0, 1].plot(time, states[1], 'r-', linewidth=2)
    axs[0, 1].set_title('Velocity Response')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Velocity (m/s)')
    axs[0, 1].grid(True)

    # Left and right chamber pressure plot (states 2,3)
    axs[1, 0].plot(time, states[2]/1000, 'g-', linewidth=2, label='Left Chamber')
    axs[1, 0].plot(time, states[3]/1000, 'm-', linewidth=2, label='Right Chamber')
    axs[1, 0].set_title('Chamber Pressure Response')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Pressure (kPa)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Control input plot
    # Fix dimension mismatch
    if control_input.ndim > 1:
        control_input = control_input.flatten()
    axs[1, 1].plot(time, control_input, 'k-', linewidth=2)
    axs[1, 1].set_title('Control Input')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Valve Position')
    axs[1, 1].grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

def plot_state_space(response):
    """
    Plot state space trajectory
    
    Parameters:
    response: Dictionary containing states
    
    Returns:
    matplotlib figure object
    """
    # Extract data
    states = response['states']
    
    # Fix state matrix dimensions
    if states.ndim == 1:
        states = states.reshape(1, -1)
    
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Check if state data is valid
    if states.shape[1] > 0:  # Ensure state data is not empty
        # Plot 3D trajectory (position, velocity, left chamber pressure)
        ax.plot3D(states[0], states[1], states[2]/1000, 'blue', linewidth=2)
        
        # Add start and end markers
        ax.scatter(states[0, 0], states[1, 0], states[2, 0]/1000, color='green', s=100, label='Start')
        ax.scatter(states[0, -1], states[1, -1], states[2, -1]/1000, color='red', s=100, label='End')
    else:
        # If data is empty, don't plot trajectory, just show a message
        ax.text(0, 0, 0, "No valid state data", fontsize=14)
    
    # Set labels and title
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_zlabel('Left Chamber Pressure (kPa)')
    ax.set_title('State Space Trajectory')

    # Add legend
    ax.legend()

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    return fig

def plot_root_locus(system):
    """
    Plot system root locus

    Parameters:
    system: Control system object

    Returns:
    matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    try:
        # Try using root_locus function to plot root locus
        # In newer versions of control library, Plot parameter is not supported
        ctrl.root_locus(system, grid=True)
        plt.title('System Root Locus')
        plt.grid(True)
    except Exception as e:
        # If an exception occurs, print error message and create a simple explanation
        print(f"Error plotting root locus: {e}")
        plt.text(0.5, 0.5, 'Cannot plot root locus\nPossible control library version issue',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, fontsize=14)
        plt.title('System Root Locus (Failed)')

    return fig

def plot_bode(system):
    """
    Plot system Bode diagram

    Parameters:
    system: Control system object

    Returns:
    matplotlib figure object
    """
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('System Bode Diagram', fontsize=16)

    try:
        # Get frequency response data using control library
        freq_resp = ctrl.frequency_response(system, np.logspace(-2, 3, 500))

        # Extract magnitude and phase data
        omega = freq_resp.omega
        mag = 20 * np.log10(np.abs(freq_resp.fresp[0][0]))
        phase = np.angle(freq_resp.fresp[0][0], deg=True)

        # Manually plot Bode diagram
        # Magnitude plot
        axs[0].semilogx(omega, mag, 'b-')
        axs[0].set_title('Magnitude Plot')
        axs[0].set_ylabel('Magnitude (dB)')
        axs[0].set_xlabel('Frequency (rad/s)')
        axs[0].grid(True, which='both')

        # Phase plot
        axs[1].semilogx(omega, phase, 'r-')
        axs[1].set_title('Phase Plot')
        axs[1].set_ylabel('Phase (deg)')
        axs[1].set_xlabel('Frequency (rad/s)')
        axs[1].grid(True, which='both')
    except Exception as e:
        # If an exception occurs, print error message and create a simple explanation
        print(f"Error plotting Bode diagram: {e}")
        axs[0].text(0.5, 0.5, 'Cannot plot Bode diagram\nPossible control library version issue',
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[0].transAxes, fontsize=14)
        axs[1].text(0.5, 0.5, 'Cannot plot Bode diagram\nPossible control library version issue',
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[1].transAxes, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_pid_response(pid_response):
    """
    Plot PID control response chart
    
    Parameters:
    pid_response: Dictionary containing PID control response
    
    Returns:
    matplotlib figure object
    """
    # Extract data
    time = pid_response['time']
    states = pid_response['states']
    output = pid_response['output']
    control_input = pid_response['input']
    setpoint = pid_response['setpoint']
    
    # Fix dimensions
    if output.ndim > 1:
        output = output.flatten()
        
    # Fix state matrix dimensions
    # Ensure state matrix is of shape (4, n)
    if states.ndim == 1:
        # If state is one-dimensional, reshape to 2D
        states = states.reshape(1, -1)
    
    # Create 3x2 subplot layout
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Position tracking plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, setpoint, 'r--', linewidth=2, label='Setpoint')
    ax1.plot(time, output, 'b-', linewidth=2, label='Actual Position')
    ax1.set_title('PID Control Position Tracking')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.legend()

    # Tracking error plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, setpoint - output, 'g-', linewidth=2)
    ax2.set_title('Tracking Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (m)')
    ax2.grid(True)

    # Control input plot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, control_input, 'k-', linewidth=2)
    ax3.set_title('Control Input (Mapped from PWM)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mapped Input (u)')
    ax3.grid(True)

    # Add secondary y-axis for PWM Duty Cycle
    ax3_twin = ax3.twinx()
    # Map u [-1, 1] to Duty Cycle [0, 100]
    duty_cycle = (control_input + 1.0) * 50.0
    ax3_twin.plot(time, duty_cycle, 'k:', linewidth=1, label='PWM Duty Cycle') # Optional: plot duty cycle as dotted line
    ax3_twin.set_ylabel('PWM Duty Cycle (%)')
    ax3_twin.set_ylim(0, 100) # Set y-axis limits for duty cycle

    # Chamber pressure plot
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time, states[2]/1000, 'g-', linewidth=2, label='Left Chamber')
    ax4.plot(time, states[3]/1000, 'm-', linewidth=2, label='Right Chamber')
    ax4.set_title('Chamber Pressure')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pressure (kPa)')
    ax4.grid(True)
    ax4.legend()

    # Velocity plot
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time, states[1], 'r-', linewidth=2)
    ax5.set_title('Velocity')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.grid(True)

    # Adjust layout
    plt.tight_layout()

    return fig
