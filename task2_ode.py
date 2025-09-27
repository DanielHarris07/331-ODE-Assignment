from functions_ode import *

# Three-Body Problem: Initial Conditions
x1 = 1.0
x1_prime = 0.0
y1 = 3.0
y1_prime = 0.0
x2 = -2.0
x2_prime = 0.0
y2 = -1.0
y2_prime = 0.0
x3 = 1.0
x3_prime = 0.0
y3 = -1.0
y3_prime = 0.0
y0 = np.array([x1,x1_prime,y1,y1_prime,x2,x2_prime,y2,y2_prime,x3,x3_prime,y3,y3_prime])

g = 1.0
m1 = 3
m2 = 4.0
m3 = 5.0
t0 = 0.0
t1 = 16
atol = 0.015

t, y = dp_solver_adaptive_step(derivative_threebody, y0, t0, t1, atol, g, m1, m2, m3)
plt.plot(y[0, :], y[2, :])
plt.plot(y[4, :], y[6, :])
plt.plot(y[8, :], y[10, :])
plt.show()
#My_Gen_AI_3BP_Animation_Tool(t,y)

