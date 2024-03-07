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
import solve_FICK_sourceTerm as mycode
import matplotlib.pyplot as plt

# Importer solution COMSOL
df = pd.read_csv('C:/Users/lucas/OneDrive/Documents/GitHub/MEC8211/devoir2/data/data_comsol.csv', delimiter=',')
r_comsol , c_comsol = df['R'], df['C']


# Calculer solutions 1er et 2e ordre
# Numerical parameters
Order = 1
n=321
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
n=321
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



plt.figure(figsize=(12, 7))  

plt.plot(r_comsol, c_comsol, 'r-', linewidth=3, label='Comsol')  
plt.plot(r_o1, c_o1, 'g--', linewidth=3, label='Solution 1er ordre')  
plt.scatter(r_o2, c_o2, color='blue', marker='o', s=100, alpha=0.7, label='Solution 2e ordre')  
plt.xlabel('Rayon (r)', fontsize=16, fontname='Helvetica')  
plt.ylabel('Concentration (C)', fontsize=16, fontname='Helvetica')  
plt.title('Concentration en fonction du Rayon, comparaison solution Comsol, 1er ordre et 2nd ordre', fontsize=18, fontweight='bold', fontname='Helvetica')  

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')  
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  

plt.tight_layout()  
plt.show()


#Calcul de l'erreur absolue

error_abs_o1 = abs(c_comsol - c_o1)
error_abs_o2 = abs(c_comsol - c_o2)

plt.figure(figsize=(10, 6))

plt.plot(r_comsol, error_abs_o1, 'g--', linewidth=2, label='Erreur abs. 1er ordre')
plt.plot(r_comsol, error_abs_o2, 'b:', linewidth=2, label='Erreur abs. 2e ordre')

plt.xlabel('Rayon (r)', fontsize=14)
plt.ylabel('Erreur absolue', fontsize=14)
plt.title('Erreur absolue entre les solutions Comsol et les solutions numériques', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()