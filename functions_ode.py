import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

def test_deriv_lecture(t,y):
    # Example ODE to check your understanding from the lecture example
    # You do not need to modify or add any code here
    return -y + t

def derivative_bungy(t, y, gravity, length, mass, drag, spring, gamma):
    """
    Compute the derivatives of the bungy jumper motion.

    Args:
        t (float): independent variable, time (s).
        y (ndarray): y[0] = vertical displacement (m), y[1] = vertical velocity (m/s).
        gravity (float): gravitational acceleration (m/s^2).
        length (float):	length of the bungy cord (m).
        mass (float): the bungy jumper's mass (m).
        drag (float): coefficient of drag (kg/m).
        spring (float): spring constant of the bungy cord (N/m).
        gamma (float): coefficient of damping (Ns/m).

    Returns:
        f (ndarray): derivatives of vertical position and vertical velocity.
    """
    f = np.array([2])
    f[0] = y[1]
    f[1] = gravity - np.sign(y[1]) * ((drag * y[1]**2) / mass) - (spring / mass) * (y[0] - length) - (gamma * y[1]) / mass
    
    return f

def explicit_rk_fixed_step(func, y0, t0, t1, h, alpha, beta, gamma, *args):
    """
    Compute solution(s) to ODE(s) using any explicit RK method with fixed step size.

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        y0 (ndarray): initial condition(s) for dependent variable(s).
        t0 (float): start value of independent variable.
        t1 (float):	stop value of independent variable.
        h (float): fixed step size along independent variable.
        alpha (ndarray): weights in the Butcher tableau.
        beta (ndarray): nodes in the Butcher tableau.
        gamma (ndarray): RK matrix in the Butcher tableau.
        *args : optional system parameters to pass to derivative function.

    Returns:
        t (ndarray): independent variable values at which dependent variable(s) calculated.
        y (ndarray): dependent variable(s) solved at t values.
    """
    # initialise independent and dependent return arrays
    t = np.array[list(range(t0, t1 + h, h))]
    tn = len(t)
    y = np.array([2, tn])
    y[:, 0] = y0
    
    # solve using RK method at each timestep
    for time in t:
        


    
    return t, y

def derivative_threebody(t,y0,g,m1,m2,m3):
    # HINT 1: One of these six function arguments will not actually be used in this threebody derivative function. However, do not remove it from the function definition, to preserve the generality of the solver functions.
    # HINT 2: For the expected format of y0, see the initial conditions provided in the task file.

    # Remember to create a docstring
    
    # TODO: Your code goes here
    pass

def dp_solver_adaptive_step(func, y0, t0, t1, atol, *args):
    """
    Compute solution to ODE using the Dormand-Prince embedded RK method with an adaptive step size.

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        y0 (ndarray): initial conditions for each solution variable.
        t0 (float): start value of independent variable.
        t1 (float):	stop value of independent variable.
        atol (float): error tolerance for determining adaptive step size.
        *args : optional system parameters to pass to derivative function.

    Returns:
        t (ndarray): independent variable values at which dependent variable(s) calculated.
        y (ndarray): dependent variable(s).
    """
    dp45_alpha = np.array([[35./384.,0.,500./1113.,125./192.,-2187./6784.,11./84.,0.], [5179./57600., 0., 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.]])
    dp45_beta = np.array([0., 1./5., 3./10., 4./5., 8./9., 1., 1.])
    dp45_gamma = np.array([[0.,0.,0.,0.,0.,0.,0.],[1./5.,0.,0.,0.,0.,0.,0.],[3./40.,9./40.,0.,0.,0.,0.,0.],[44./45.,-56./15.,32./9.,0.,0.,0.,0.],[19372./6561.,-25360./2187.,64448./6561.,-212./729.,0.,0.,0.],[9017./3168.,-355./33.,46732./5247.,49./176.,-5103./18656.,0.,0.],[35./384.,0.,500./1113.,125./192.,-2187./6784.,11./84.,0.]],)
    
    # TODO: Your code goes here
    pass

def My_Gen_AI_3BP_Animation_Tool(t,y):
    """
    An animation plotting tool produced by generative AI
    DO NOT MODIFY OR ADD ANY CODE IN THIS FUNCTION
    """
    
    step_lengths = np.diff(t)
    step_times = t[1:]

    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 8),
    gridspec_kw={'height_ratios': [3, 1]}  # Motion plot gets 3x the space
    )
    ax1.set_aspect('equal', adjustable='box')
    fig.tight_layout(pad=4)

    # Subplot 1: 2D motion
    colors = ['red', 'green', 'blue']
    scatters = [ax1.plot([], [], 'o', color=c, markersize=8)[0] for c in colors]
    trails = [ax1.plot([], [], '-', color=c, alpha=0.6)[0] for c in colors]

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_title("2D Motion of Bodies")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Subplot 2: Timestep length plot
    line, = ax2.plot([], [], lw=2, color='purple')
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, max(step_lengths) * 1.2)
    ax2.set_title("Timestep Length Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Step Length (s)")

    # Animation update function
    def update(frame):
        for i in range(3):
            x = y[4 * i][frame]
            y_coord = y[4 * i + 2][frame]
            scatters[i].set_data([x], [y_coord])

            # Update trail
            trails[i].set_data(y[4 * i][:frame + 1], y[4 * i + 2][:frame + 1])

        # Update timestep plot
        if frame > 0:
            line.set_data(step_times[:frame], step_lengths[:frame])

        return scatters + trails + [line]

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(t), interval=100, blit=True)

    plt.show()

    return