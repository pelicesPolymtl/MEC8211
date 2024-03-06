'''
##################################################
## Run simulations (using olve_FICK_sourceTerm.py)
##################################################
## Code to MEC8211
##################################################
## Authors:
##     - Pablo ELICES PAZ
##     - Lucas BRAHIC
##     - Justin BELZILE
## Date: 04/03/2024
##################################################
## Description:
##    This script is used to debug the solution
##    It plot one solution for a: 
##     - tmax: final time
##     - n: number of point of the mesh
##     - order: order of the finite differences
##     - dt: time step
##################################################
'''


import numpy as np
import solve_FICK_sourceTerm as mycode
import matplotlib.pyplot as plt

# Numerical parameters
Order = 2
n=20
tMax = 1e9
dt = 1E7


tVer = tMax/10

print('runing solver...')
print(' order=', Order)
print('')



t = 0
factor_t = 1
r,c = mycode.solve(n, dt, Order, 0, MMS = False, debug=False)
plt.plot(r,c, label='t = '+str(int(t/factor_t)))
while(t<tMax):
    t += tVer
    print('time t=', t)
    r,c = mycode.solve(n, dt, Order, t, MMS = False, debug=False)
    plt.plot(r,c, label='t = '+"{:.1e}".format(t))
plt.title('Unsteady solution. N='+str(n)+'; dt='+"{:.1e}".format(dt) ,
          fontsize=14, fontweight='bold', y=1.02)
plt.xlabel('Position (r)')
plt.ylabel('Concentration (C)')
plt.legend()
plt.savefig('unsteady.png')


