##################################################
## Run simulations (using solve_FICK.py)
##################################################
## Code to MEC8211
##################################################
## Authors: 
##     - Pablo ELICES PAZ
##     - 
##     - 
## Date: 06/02/2024
##################################################
## ToDo: 
##     - Check 2nd order solution
##     - Plot convergence
##################################################

import numpy as np
from numpy import linalg as LA 
import solve_FICK
import time
import matplotlib.pyplot as plt



def plot_solutions(N_cases, order, file_name):

    # Numerical parameters
    dt = 1E5
    tol = 1E-15
    imax = 1E8

    print('runing solver...')
    print(' order=', order)
    print('')

    for N in N_cases:
        print(N)
        r, numSol, anSol = solve_FICK.solve(N+1, dt, order, imax, tol, debug = False)
        print('N, errorL2,',N,  LA.norm(numSol-anSol)*np.sqrt(1/N))
        plt.plot(r,numSol, label='N='+str(N))
    plt.title('Order: '+str(order))
    plt.legend()
    plt.xlabel("r [m]")
    plt.ylabel("C [mol/m³]")
    plt.savefig(file_name)

def plot_convergence(N_cases, order):
    return 0



time0 = time.time()

# First order plot solution
N_cases = [ 5, 10, 20, 40, 80]
order = 1
plot_solutions(N_cases, order, file_name='solution_1stOrder.png')
plot_convergence(N_cases, order)

# Second order plot solution 
N_cases = [ 5, 10, 20, 40, 80]
order = 2
plot_solutions(N_cases, order, file_name='solution_2ndOrder.png')
plot_convergence(N_cases, order)

print('time required: ', time.time() - time0)


