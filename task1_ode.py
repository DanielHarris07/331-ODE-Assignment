from functions_ode import *
import matplotlib.pyplot as plt

test = False
if test: 
    # Test case: example from lecture slides
    # You can remove this section of code from your submission file once you are confident your functions are working properly
    RK4_38_alpha = np.array([1./8., 3./8., 3./8., 1./8.])
    RK4_38_beta = np.array([0., 1./3., 2./3., 1.])
    RK4_38_gamma = np.array([[0.,0.,0.,0.],[1./3.,0.,0.,0.],[-1./3.,1.,0.,0.],[1.,-1.,1.,0.]])
    test_t, test_y = explicit_rk_fixed_step(test_deriv_lecture,np.array([4]),0,2,2, RK4_38_alpha, RK4_38_beta, RK4_38_gamma)
    print(test_y)
    print(np.shape(test_y))


# bungy parameters
t0 = 0
t1 = 50
h = 0.1
y0 = np.array([0., 2.])
gravity = 9.8
drag = 0.75
gamma = 8.0
mass = 67.0

# Butcher table: Classic RK4 explicit method
rk4_alpha = np.array([1./6., 1./3., 1./3., 1./6.])
rk4_beta = np.array([0., 1./2., 1./2., 1.])
rk4_gamma = np.array([[0., 0., 0., 0.], [1./2., 0., 0., 0.], [0., 1./2., 0., 0.], [0., 0., 1., 0.]])

# bungy cords
spring = np.array(list(range(50, 101, 10)) * 2)
length = np.array([15] * 6 + [20] * 6)
solutions = [None] * 12
for i in range(12):
    t, solutions[i] = explicit_rk_fixed_step(derivative_bungy, y0, t0, t1, h,
                                                rk4_alpha, rk4_beta, rk4_gamma, gravity, length[i], mass, drag, spring[i], gamma)

plt.plot(t, solutions[2][1, :])
plt.show()