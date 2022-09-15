import time
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

print("2D wave equation solver")

def SolveHeatEquation(max_iter_time, plate_length, boundary_conditions, gamma, initial_condition=0):


# Initialize solution: the grid of u(k, i, j)
    u = np.empty((max_iter_time, plate_length))

# Set the initial condition
    u.fill(u_initial)

# Set the boundary conditions
    u[:, :(plate_length-1)] = boundary_conditions[0]
    u[:, (plate_length-1):] = boundary_conditions[1]

    def calculate(u):
        for k in range(0, max_iter_time-1, 1):
            for i in range(1, plate_length-1, delta_x):
                    u[k + 1, i] = gamma * (u[k][i+1] + u[k][i-1] - 2*u[k][i]) + u[k][i]
        return u

    def plotheatmap(u_k, k):
    # Clear the current plot figure
        plt.clf()

        plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
        plt.xlabel("x")
        plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
        plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
        plt.colorbar()

        return plt

# Do the calculation here
    return calculate(u)
    
    def animate(k):
        plotheatmap(u[k], k)


    # anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
    # anim.save("wave_equation_solution.gif")

    print("Done!")

plate_length = 50
max_iter_time = 750

alpha = 2
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
Gamma = (alpha * delta_t) / (delta_x ** 2)

# Boundary conditions
u_top = 100.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0
bc = [u_bottom, u_top]

# Initial condition everywhere inside the grid
u_initial = 0

u = SolveHeatEquation(max_iter_time, plate_length, bc, Gamma)
plt.plot(u[-1])
# plt.savefig('finitediff.png')
end_time = time.time()

print("Total execution time : " + str(end_time - start_time) + " seconds")

#Method of lines
#--------------------------X----------------------
import time
from matplotlib import pyplot as plt
# %matplotlib inline

start_time = time.time()
manual_seed = 123

import pde
grid = pde.UnitGrid([20])                 # generate grid
state = pde.ScalarField.random_uniform(grid) # generate initial condition
bc = [{"value" : "0"},{"derivative" : "0"}] #Try and understand the initial and boundary conditions.
# bc_y = [{"value" : "100"}]
# bc = [bc_x,bc_y]
state.laplace(bc)  # enforces boundary condtion'

eq = pde.PDE({"u":"0.1*laplace(u)"}) #define the PDE right hand side only

result = eq.solve(state, t_range=1000, dt=1e-3) #solve it

result.plot() # plot the result
# pde.visualization.plotting.savefig('py-pde.png')
# plt.savefig('py-pde.png', bbox_inches='tight')

end_time = time.time()

print("Total execution time : " + str(end_time - start_time) + " seconds")

# Neural Network 
#-------------------X-----------------------------
import time
start_time = time.time()


from pydens.model_torch import Solver, D
from pydens.batchflow import NumpySampler as NS
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from pydens.model_torch import V

def pde(v,x,t):
  return D(v, t) - D(D(v, x), x) #heat equation

solver = Solver(equation=pde, ndims=2,initial_condition = 0 ,boundary_condition = 0)

solver.fit(batch_size=30, niters=100, lr=0.05)

plt.plot(solver.losses[:])
ans=solver.predict(np.array([.75, .25, 0]), np.array([.75, .25, 1]))

# plt.savefig('pydens.png')

end_time = time.time()

print("Total execution time : " + str(end_time - start_time) + " seconds", "Solution for the equation-",ans)