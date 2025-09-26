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
print(np.shape(y0))
[x1, y1, x2, y2, x3, y3] = y0[0:12:2]

g = 1.0
m1 = 3.0
m2 = 4.0
m3 = 5.0
t0 = 0.0
t1 = 18.0
tol = 0.015

f = derivative_threebody(None, y0, g, m1, m2, m3)
print(f)
#My_Gen_AI_3BP_Animation_Tool(t,y)

