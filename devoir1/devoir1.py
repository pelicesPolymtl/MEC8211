##################################################
## Run simulations (using solve_FICK.py)
##################################################
## Code to MEC8211
##################################################
## Authors: 
##     - Pablo ELICES PAZ
##     - Lucas BRAHIC
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
    error_L1 = []
    error_L2 = []
    error_Linfini = []
    print('runing solver...')
    print(' order=', order)
    print('')

    for N in N_cases:
        print(N)
        r, numSol, anSol = solve_FICK.solve(N+1, dt, order, imax, tol, debug = False)
        errorL1 = LA.norm(numSol-anSol)*1/N
        errorL2 = LA.norm(numSol-anSol)*np.sqrt(1/N)
        errorLinfini = np.max(np.abs(numSol - anSol))
        print('N, errorL1,',N,  errorL1)
        print('N, errorL2,',N,  errorL2)
        print('N, errorLinfini,',N,  errorLinfini)
        error_L1.append(errorL1)
        error_L2.append(errorL2)
        error_Linfini.append(errorLinfini)
        plt.plot(r,numSol, label='N='+str(N))
    

    plt.title('Order: '+str(order))
    plt.legend()
    plt.xlabel("r [m]")
    plt.ylabel("C [mol/m³]")
    plt.savefig(file_name)
    plt.show()
    return error_L1,error_L2,error_Linfini

time0 = time.time()

# First order plot solution
N_cases = [ 5, 10, 20, 40, 80]
order = 1
errors_values_1_L1,errors_values_1_L2,errorLinf_1 = plot_solutions(N_cases, order, file_name='solution_1stOrder.png')

# Second order plot solution 
N_cases = [ 5, 10, 20, 40, 80]
order = 2
errors_values_2_L1,errors_values_2_L2,errorLinf_2 = plot_solutions(N_cases, order, file_name='solution_2ndOrder.png')

print('time required: ', time.time() - time0)



## Convergence ordre 2 erreur L2 en fonction de delta_x

h_values = [0.5/i for i in N_cases]   #résolution 
h_values.reverse()
errorLinf_1.reverse()
errorLinf_2.reverse()
errors_values_1_L1.reverse()
errors_values_2_L1.reverse()
errors_values_1_L2.reverse()
errors_values_2_L2.reverse()

def plot_erreurs(errors_valuesL1,errors_valuesL2,errorLinf,order):
    plt.figure(figsize=(10, 6))
    plt.plot(h_values, errorLinf, label='Erreur Linfini', color='blue', linewidth=2, marker='o')
    plt.plot(h_values, errors_valuesL1, label='Erreur L1', color='red', linestyle='--', marker='x')
    plt.plot(h_values, errors_valuesL2, label='Erreur L2', color='green', linestyle='-.', marker='s')
    plt.xlabel('Maillage en m^-1')
    plt.ylabel('Erreurs en mol/m^3')
    plt.title('Comparaison des erreurs pour différentes résoltuions, pour ordre' + order)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_erreurs(errors_values_1_L1,errors_values_1_L2,errorLinf_1, "1")
#plot_erreurs(errors_values_2_L1,errors_values_2_L2,errorLinf_2, "1")




def plot_convergence(error_values, h_values, order) :
    

    coefficients = np.polyfit(np.log(h_values[:3]), np.log(error_values[:3]), 1)
    exponent = coefficients[0]

    fit_function_log = lambda x: exponent * x + coefficients[1]

    fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

    extrapolated_value = fit_function(h_values[-1])

    plt.figure(figsize=(8, 6))
    plt.scatter(h_values, error_values, marker='o', color='b', label='Données numériques obtenues')
    plt.plot(h_values, fit_function(h_values), linestyle='--', color='r', label='Régression en loi de puissance')

    plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

    plt.title('Convergence d\'ordre' + order + '\n de l\'erreur $L_2$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

    plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (cm)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
    plt.ylabel('Erreur $L_2$ (m/s)', fontsize=12, fontweight='bold')

    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

    equation_text = f'$L_2 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
    equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

    equation_text_obj.set_position((0.5, 0.4))


    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
    
plot_convergence(errors_values_1_L2, h_values, "1")
#plot_convergence(errors_values_2_L2, h_values, "2")




