'''
##################################################
## 
##################################################
## Code to MEC8211
##################################################
## Authors:
##     - Pablo ELICES PAZ
##     - Lucas BRAHIC
##     - Justin BELZILE
## Date: 06/03/2024
##################################################
## Description:
##    This script is used to plot the L2 error
##    between the 1st and 2nd ordre of the 
##    solution and a solution from COMSOL
##################################################
'''

import pandas as pd
#import numpy as np
import solve_FICK_sourceTerm as mycode
#from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Importer solution COMSOL
df = pd.read_csv('C:/Users/pc/Downloads/devoir2/data/data_comsol.csv', delimiter=',')
r_comsol , c_comsol = df['R'], df['C']


# Calculer solutions 1er et 2e ordre
# Numerical parameters
Order = 1
n=20
tMax = 1e11
dt = 1E7
tVer = tMax/10

print('runing solver...')
print(' order=', Order)
print('')

t = 0
factor_t = 1e7
r_o1,c_01 = mycode.solve(n, dt, Order, 0, MMS = False, debug=False)
while(t<tMax):
    t += tVer
    print('time t=', t)
    r_o1,c_o1 = mycode.solve(n, dt, Order, t, MMS = False, debug=False)

Order = 2
n=20
tMax = 1e11
dt = 1E7
tVer = tMax/10

print('runing solver...')
print(' order=', Order)
print('')

t = 0
factor_t = 1e7
r_o2,c_o2 = mycode.solve(n, dt, Order, 0, MMS = False, debug=False)

while(t<tMax):
    t += tVer
    print('time t=', t)
    r_o2,c_o2 = mycode.solve(n, dt, Order, t, MMS = False, debug=False)

# Plot les 3 
plt.plot(r_comsol, c_comsol, color='red', linestyle='-', linewidth=1, label='Comsol')
plt.plot(r_o1, c_o1, color='green', linestyle='-', linewidth=2, label='Solution 1er ordre')
plt.scatter(r_o2, c_o2, color='blue', marker='o', label='Solution 2e ordre')

# Titres, légendes, etc.
plt.xlabel('Rayon')
plt.ylabel('Concentration')
plt.title('Concentration en fonction du Rayon')
plt.legend()
plt.grid(True)
plt.show()