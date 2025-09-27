from functions_ode import *
y0 = np.array([1, 0])
t0 = 0
t1 = 20
ha = 0.1
hb = 0.01
tol = 1e-3
max_iter = 10
mu = 0.

t, y = backward_euler_solver(derivative_vanderpol, y0, t0, t1, ha, tol, max_iter, mu)
plt.plot(y[0, :], y[1, :])
plt.xlabel("y")
plt.ylabel("dy/dt")
plt.title("Step Size = 0.1")
plt.show()

t, y = backward_euler_solver(derivative_vanderpol, y0, t0, t1, hb, tol, max_iter, mu)
plt.plot(y[0, :], y[1, :])
plt.xlabel("y")
plt.ylabel("dy/dt")
plt.title("Step Size = 0.01")
plt.show()

t, y = backward_euler_solver(derivative_vanderpol, y0, t0, t1, hb, tol, max_iter, 0)
plt.plot(y[0, :], y[1, :], label="μ = 0")

t, y = backward_euler_solver(derivative_vanderpol, y0, t0, t1, hb, tol, max_iter, 2)
plt.plot(y[0, :], y[1, :], label="μ = 2")

t, y = backward_euler_solver(derivative_vanderpol, y0, t0, t1, hb, tol, max_iter, 4)
plt.plot(y[0, :], y[1, :], label="μ = 4")


plt.title("Phase Plot of Van der Pol Oscillator for Varying Values of μ")
plt.xlabel("y")
plt.ylabel("dy/dt")
plt.legend()
plt.show()