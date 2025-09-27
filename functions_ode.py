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
    f = np.zeros([2, 1])
    f[0] = y[1]
    
    # If bungee is still in freefall, remove other terms
    if y[0] < length:
        spring = 0
        gamma = 0
    f[1] = (mass * gravity - np.sign(y[1])*drag*y[1]**2 - spring*(y[0]  - length) - gamma * y[1]) / mass
    
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
    tn = int(np.floor((t1 - t0)/h) + 1)
    t = np.array(np.linspace(t0, t1, tn))
    yn = len(y0)
    y = np.zeros([yn, tn]) 
    fn = len(alpha)
    # set initial conditions for the dependent variables
    y[:, 0] = y0
    
    # solve using RK method at each timestep
    for i in range(tn - 1): 
        f = np.zeros([yn, fn]) # derivatives at each time satep
        for j in range(fn): 
            f_current = f @ gamma[j, :][:, np.newaxis] # f weights used for the current beta step
            y_new = (y[:, i:i+1] + h * f_current).flatten()
            t_new = t[i] + h*beta[j]
            f[:, j:j+1] = func(t_new, y_new, *args)

        # update y matrix with values calculated at the new timestep according to alpha weightings on all yfn derivatives
        y[:, i+1:i+2] = y[:, i:i+1] + h * f @ alpha[:, np.newaxis]

    return t, y

def derivative_threebody(t,y0,g,m1,m2,m3):
    """
    Compute the derivates of the three-body-problem in two dimensions (x-y plane).

    Args:
        t (float): independent variable, time (s).
        y0 (ndarray): position (m) and velocity (m/s) of each body in each plane
        g (float): gravitational acceleration (m/s^2).
        m1 (float): masss of first body (kg)
        m2 (float): masss of second body (kg)
        m3 (float): masss of third body (kg)

    Returns:
        f (ndarray): velocity (m/s) and acceleration (m/s^2) of each body in each plane:
    """
    f = np.zeros([12])
    [x1, y1, x2, y2, x3, y3] = y0[0:12:2] 
    # assign velocity values from input directly to output vector
    f[0:11:2] = y0[1:12:2] 

    # calculate accerations from position inputs
    f[1:4:2] = (np.array([(x2 - x1), (y2- y1)]) * (g*m2/ ((x1-x2)**2 + (y1-y2)**2)**(3/2)) + #body 1
    np.array([(x3 - x1), (y3- y1)]) * (g*m3/ ((x1-x3)**2 + (y1-y3)**2)**(3/2)))

    f[5:8:2] = (np.array([(x1 - x2), (y1- y2)]) * (g*m1/ ((x1-x2)**2 + (y1-y2)**2)**(3/2)) + #body 2
    np.array([(x3 - x2), (y3- y2)]) * (g*m3/ ((x2-x3)**2 + (y2-y3)**2)**(3/2)))

    f[9:12:2] = (np.array([(x1 - x3), (y1- y3)]) * (g*m1/ ((x1-x3)**2 + (y1-y3)**2)**(3/2)) + #body 3
    np.array([(x2 - x3), (y2- y3)]) * (g*m2/ ((x2-x3)**2 + (y2-y3)**2)**(3/2)))

    return f


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
    safety_factor = 0.9
    min_step = 1e-7
    h = 0.01 # initial stedp guess from brief
    # set initial conditions for the dependent variables
    y = y0[:, np.newaxis]
    
    # solve using dp45 method at each timestep
    t = [t0]
    i = 0
    while t[-1] < t1:
        # Check the current step wont take us past t1
        if t[-1] + h > t1:
            h = t1 - t[-1]
        
        f = np.zeros([12, 7]) # derivatives at each time step (should be 12 x 7)
        for j in range(7): 
            f_current = f @ dp45_gamma[j, :][:, np.newaxis] # f weights used for the current beta step
            y_new = (y[:, i] + h * f_current).flatten()
            t_new = t[-1] + h*dp45_beta[j]
            f[:, j] = func(t_new, y_new, *args) 

        # update y matrix with values calculated at the new timestep according to alpha weightings on all yfn derivatives
        y_5th = y[:, i:i+1] + h * f @ dp45_alpha[0, :][:, np.newaxis]
        y_4th = y[:, i:i+1] + h * f @ dp45_alpha[1, :][:, np.newaxis]

        # check error on the current step size and adjust accordingly
        
        system_err = np.abs(y_5th - y_4th) / atol
        max_err = np.max(system_err)
        max_i = np.argmax(system_err)
        # If step was good or at minimum already
        if (max_err <= atol) or (h == min_step):
            y = np.concatenate((y, y_5th), axis=1)
            t = np.append(t, t[-1] + h)
            i += 1
            h = safety_factor * h * (atol / system_err[max_i])**(1./5.)
        else: # if step was bad
            h = safety_factor * h * (atol / system_err[max_i])**(1./5.)

        if h < min_step:
            h = min_step
        print(f'Max Error: {max_err}  Step Size: {h}  Iteration: {i}  Time Step: {len(t)}  Time: {t[-1]}')

    return t, y

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

def derivative_vanderpol(t, y, mu):
    """
    Compute the derivatives of the Van der Pol oscillator

    Args:
        t (float): independent variable, time (s).
        y (ndarray): y[0] = y, y[1] = y'.
        mu (float): strength of non-linear damping.
        
    Returns:
        f (ndarray): f[0] = y', f[1] = y''.
    """
    f = np.zeros([2])
    f[0] = y[1]
    f[1] = mu*(1.-y[0]**2)*y[1] - y[0]

    return f

def backward_euler_solver(func, y0, t0, t1, h, tol, max_iter, *args):
    '''
    Solves an ODE using the implicit backward euler method

    Args: 
        func (callable): ODE function
        y0 (ndarray): initial conditions of the system
        t0 (float): initial time
        t1 (float): final time
        h (float): step size
        tol (float): convergence tolerance for implicit method
        max_iter (int): backup convergence tolerance for implicit method
        *args : optional system parameters to pass to ODE function
    '''
     # initialise independent and dependent return arrays
    tn = int(np.floor((t1 - t0)/h) + 1)
    t = np.array(np.linspace(t0, t1, tn))
    yn = len(y0)
    y = np.zeros([yn, tn]) 
    # set initial conditions for the dependent variables
    y[:, 0] = y0

    for i in range(tn - 1):
        #perform initial guess using forward Euler
        y_guess = y[:, i] + h * func(t[i], y[:, i], *args)
        for j in range(max_iter):
            y_next = y[:, i] + h * func(t[i] + h, y_guess, *args)
            # check if converged
            abs_err = (y_next - y_guess) / (1 + y_next)
            if all(abs_err < tol):
                break
            else:
                y_guess = y_next
        y[:, i + 1] = y_next
            
    return t, y