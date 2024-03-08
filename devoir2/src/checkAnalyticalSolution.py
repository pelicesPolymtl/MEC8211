'''
##################################################
## Check analytical solution
##################################################
## Code to MEC8211
##################################################
## Authors:
##     - Pablo ELICES PAZ
##     - Lucas BRAHIC
##     - Justin BELZILE
## Date: 07/03/2024
##################################################
## Description:
##    This script is used plot the solution for
##    Fick's equation over a specified time range
##################################################
'''
import numpy as np
import matplotlib.pyplot as plt
import solve_FICK_sourceTerm as mycode

# Time range
n = 20
tFinal = 15

# Loop over timestep
for t in np.linspace(0,tFinal,n+1):
    # Geometry
    r0 = 0
    rf = 0.5
    r = np.linspace(r0, rf, n)
    c_an = mycode.manufactured_solution(r, t)
    plt.plot(r, c_an, label='time t='+str(t))

plt.legend()
plt.show()
