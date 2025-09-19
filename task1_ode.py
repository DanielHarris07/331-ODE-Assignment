from functions_ode import *


# Test case: example from lecture slides
# You can remove this section of code from your submission file once you are confident your functions are working properly
RK4_38_alpha = np.array([1./8., 3./8., 3./8., 1./8.])
RK4_38_beta = np.array([0., 1./3., 2./3., 1.])
RK4_38_gamma = np.array([[0.,0.,0.,0.],[1./3.,0.,0.,0.],[-1./3.,1.,0.,0.],[1.,-1.,1.,0.]])
test_t, test_y = explicit_RK_fixed_step(test_deriv_lecture,np.array([4]),0,2,2, RK4_38_alpha, RK4_38_beta, RK4_38_gamma)
print(test_y)

# bungy parameters
t0 = 0
t1 = 50
h = 0.1
y0 = np.array([0., 2.])
gravity = 9.81
drag = 0.75
gamma = 8.0
mass = 67.0

# Butcher table: Classic RK4 explicit method
rk4_alpha = np.array([1./6., 1./3., 1./3., 1./6.])
rk4_beta = np.array([0., 1./2., 1./2., 1.])
rk4_gamma = np.array([[0., 0., 0., 0.], [1./2., 0., 0., 0.], [0., 1./2., 0., 0.], [0., 0., 1., 0.]])

# TODO - your code here