import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
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
    # get (x, y) position-tuples from the input dependent variable vector
    r1 = np.array([y0[0], y0[2]])
    r2 = np.array([y0[4], y0[6]])
    r3 = np.array([y0[8], y0[10]])
    # calculate sum of forces acting on each body with respect to each other body
    f1 = g*m2*(r2 - r1) / ((np.linalg.norm(r1 - r2))**3) + g*m3*(r3 - r1) / ((np.linalg.norm(r1 - r3))**3)
    f2 = g*m1*(r1 - r2) / ((np.linalg.norm(r2 - r1))**3) + g*m3*(r3 - r2) / ((np.linalg.norm(r2 - r3))**3)
    f3 = g*m1*(r1 - r3) / ((np.linalg.norm(r3 - r1))**3) + g*m2*(r2 - r3) / ((np.linalg.norm(r3 - r2))**3)
    f = np.array([y0[1], f1[0], y0[3], f1[1], y0[5], f2[0], y0[7], f2[1], y0[9], f3[0], y0[11], f3[1]])
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
    h_new = 0.01 # initial stedp guess from brief
    # set initial conditions for the dependent variables
    y = y0[:, np.newaxis]

    # solve using dp45 method at each timestep
    t = [t0]
    while t[-1] < t1:
        # Check the current step wont take us past t1
        if t[-1] + h_new > t1:
            h_new = t1 - t[-1]
        
        f = np.zeros([12, 7]) # derivatives at each time step (should be 12 x 7)
        for j in range(7): 
            f_current = f @ dp45_gamma[j, :][:, np.newaxis] # f weights used for the current beta step
            y_new = (y[:, -1:] + h_new * f_current).flatten()
            t_new = t[-1] + h_new * dp45_beta[j]
            f[:, j] = func(t_new, y_new, *args) 
            
        # update y matrix with values calculated at the new timestep according to alpha weightings on all yfn derivatives
        y_5th = h_new * f @ dp45_alpha[0, :][:, np.newaxis]
        y_4th = h_new * f @ dp45_alpha[1, :][:, np.newaxis]

        # check error on the current step size and adjust accordingly
        max_system_err = max(abs(y_5th - y_4th)/ atol)

        # If step was good or at minimum already
        if max_system_err < 1 or h_new == min_step:
            y = np.append(y, (y_5th + y[:, -1:]), axis=1) 
            t = np.append(t, t[-1] + h_new)
        
        h_old = h_new
        h_new = safety_factor * h_old * (1/max_system_err)**(1/5)

        if h_new < min_step:
            h_new = min_step
        
        print(f'Max Error: {max_system_err}  Step Size: {h_old} Time Step: {len(t)}  Time: {t[-1]}')

    return t, y