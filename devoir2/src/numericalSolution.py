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
dt = 1E5
Order = 2

print('runing solver...')
print(' order=', Order)
print('')

n=20


tMax = 1e8
tVer = tMax/10

t = 0
factor_t = 1e7
r,c = mycode.solve(n, dt, Order, 0, MMS = False, debug=False)
plt.plot(r,c, label='t = '+str(int(t/factor_t)))
while(t<tMax):
    t += tVer
    print('time t=', t)
    r,c = mycode.solve(n, dt, Order, t, MMS = False, debug=False)
    plt.plot(r,c, label='t = '+str(int(t/factor_t)))
plt.legend()
plt.show()

