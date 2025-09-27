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


# solve for all 12 bungee cords
spring = np.array(list(range(50, 101, 10)) * 2)
length = np.array([15] * 6 + [20] * 6)
solutions = [None] * 12
for i in range(12):
    t, solutions[i] = explicit_rk_fixed_step(derivative_bungy, y0, t0, t1, h,
                                                rk4_alpha, rk4_beta, rk4_gamma, gravity, length[i], mass, drag, spring[i], gamma)

# plot all bungee cords max displacement
max_displacement = np.zeros([12])
labels = [None] * 12
for i in range(12):
    labels[i] = f"{"SHORT" if i < 6 else "REG"}{spring[i]}"
    max_displacement[i] = max(solutions[i][0, :])

bar_colors = ['tab:blue'] * 12 


fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(labels, max_displacement, color=bar_colors)
plt.axhline(y=43, color='red', linestyle='--', linewidth=2, label='43 m')
ax.set_ylabel('Maximum Displacement [m]')
ax.set_title('Maximum Displacement by Cord Length and Stiffness')
ax.legend()
plt.show()

# Plot best cord
plt.plot(t, solutions[8][0, :], label="Vertical Displacment [m]")
plt.plot(t, solutions[8][1, :], label="Velocity [m/s]")
plt.xlabel("Time [s]")
plt.title("Vertical Displacement and Velocity against Time for REG70 Bungee Cord")
plt.show()

plt.plot(solutions[8][0, :], solutions[8][1, :])
plt.xlabel("Vertical Displacement [m]")
plt.ylabel("Vertical Velocity [m/s]")
plt.title("Phase Plot of Vecrtical Velocity against Displacement for REG70 Bungee Cord")
plt.show()

# Find 41.2m impact velocity with REG70
idx = (np.abs(solutions[8][0, :] - 41.2)).argmin()
print(f'Velocity = {solutions[8][1, idx]}')

# jumper is actuall 85kg
mass = 85
t, solution = explicit_rk_fixed_step(derivative_bungy, y0, t0, t1, h,
                                                rk4_alpha, rk4_beta, rk4_gamma, gravity, length[8], mass, drag, spring[8], gamma)
# Plot best cord
plt.plot(t, solution[0, :], label="Vertical Displacment [m]")
plt.plot(t, solution[1, :], label="Velocity [m/s]")
plt.xlabel("Time [s]")
plt.title("Vertical Displacement and Velocity against Time for REG70 85kg Jumper")
plt.show()
print(f'Max Displacement: {np.max(solution[0, :])}')
idx = (np.abs(solution[0, :] - 41.2)).argmin()
print(f'Impact Velocity: {solution[1, idx]}')