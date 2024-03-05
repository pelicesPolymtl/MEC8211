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
## ToDo:
##     -
##################################################
'''


import numpy as np
import solve_FICK_sourceTerm as mycode
import matplotlib.pyplot as plt

# Numerical parameters

Order = 2

print('runing solver...')
print(' order=', Order)
print('')

n=20
tMax = 0.1

dt = tMax/2000





print('time t=', tMax)
r,c = mycode.solve(n, dt, Order, tMax, MMS = True, debug=False)
c_an = mycode.manufactured_solution(r, tMax)
plt.plot(r, c, label='numeric')
plt.plot(r, c_an, label='analytical')
plt.legend()
plt.show()


