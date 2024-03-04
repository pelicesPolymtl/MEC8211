'''
##################################################
## Run simulations (using solve_fick.py)
##################################################
## Code to MEC8211
##################################################
## Authors:
##     - Pablo ELICES PAZ
##     - Lucas BRAHIC
##     - Justin BELZILE
## Date: 11/02/2024
##################################################
## ToDo:
##     -
##################################################
'''

import time
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import solve_FICK as solve_fick
# import untitled0 as solve_fick
def plot_solutions(n_cases_ext, Order, file_name):
    '''
    Plots the solutions for different cases.

    Args:
        n_cases_ext (list): List of integers representing the number of cases to plot.
        order (int): The order of the solutions (1 or 2).
        file_name (str): The name of the file to save the plot.

    Returns:
        tuple: A tuple containing the error values for each case.
    '''
    # Numerical parameters
    dt = 1E5
    tol = 1E-15
    imax = 1E8
    error_l_1 = []
    error_l_2 = []
    error_l_infini = []
    print('runing solver...')
    print(' order=', Order)
    print('')

    for n in n_cases_ext:
        print(n)
        r, num_sol, an_sol = solve_fick.solve(n+1, dt, Order, imax, tol, debug = False)
        error_l1 = LA.norm(num_sol-an_sol)*1/n
        error_l2 = LA.norm(num_sol-an_sol)*np.sqrt(1/n)
        error_linfini = np.max(np.abs(num_sol - an_sol))
        print('N, errorL1,',n,  error_l1)
        print('N, errorL2,',n,  error_l2)
        print('N, errorLinfini,',n,  error_linfini)
        error_l_1.append(error_l1)
        error_l_2.append(error_l2)
        error_l_infini.append(error_linfini)
        plt.plot(r,num_sol, label='N='+str(n))

    plt.title('Order: '+str(Order))
    plt.legend()
    plt.xlabel("r [m]")
    plt.ylabel("C [mol/m³]")
    plt.savefig(file_name)
    plt.show()
    return error_l_1,error_l_2,error_l_infini

time0 = time.time()

# First order plot solution
n_cases = [ 5, 10, 20, 40, 80]
Order = 1
errors_values_1_l1,errors_values_1_l2,error_linf_1 = plot_solutions(
    n_cases, Order, file_name='solution_1stOrder.png')

# Second order plot solution
n_cases = [ 5, 10, 20, 40, 80]
Order = 2
errors_values_2_l1,errors_values_2_l2,error_linf_2 = plot_solutions(
    n_cases, Order, file_name='solution_2ndOrder.png')

print('time required: ', time.time() - time0)

## Convergence ordre 2 erreur L2 en fonction de delta_x

h_values = [0.5/i for i in n_cases]   #résolution
h_values.reverse()
error_linf_1.reverse()
error_linf_2.reverse()
errors_values_1_l1.reverse()
errors_values_2_l1.reverse()
errors_values_1_l2.reverse()
errors_values_2_l2.reverse()

def plot_erreurs(errors_values_l1,errors_values_l2,error_linf,Order):
    '''
    Plots the errors for different resolutions.

    Args:
        errors_values_l1 (list): List of L1 errors for each resolution.
        errors_values_l2 (list): List of L2 errors for each resolution.
        errorLinf (list): List of Linf errors for each resolution.
        order (str): The order of the solutions ('1' or '2').

    Returns:
        None
        '''
    plt.figure(figsize=(10, 6))
    plt.plot(h_values, error_linf, label='Erreur l_infini',
             color='blue', linewidth=2, marker='o')
    plt.plot(h_values, errors_values_l1, label='Erreur l_1',
             color='red', linestyle='--', marker='x')
    plt.plot(h_values, errors_values_l2, label='Erreur l_2',
             color='green', linestyle='-.', marker='s')
    plt.xlabel('Maillage en m^-1')
    plt.ylabel('Erreurs en mol/m^3')
    plt.title('Comparaison des erreurs pour différentes résoltuions, pour ordre' + Order)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_erreurs(errors_values_1_l1,errors_values_1_l2,error_linf_1, "1")
plot_erreurs(errors_values_2_l1,errors_values_2_l2,error_linf_2, "1")

def plot_convergence(error_values, h_values_ext, Order, error_name = 'L2') :
    """
    Plots the convergence of the error with respect to the grid size.

    Args:
        error_values (list): List of error values.
        h_values_ext (list): List of grid sizes.
        order (int): The order of the convergence (1 or 2).
        error_name (string): name of the error (L1, L2, Linf)

    Returns:
        None
    """

    coefficients = np.polyfit(np.log(h_values_ext[:3]), np.log(error_values[:3]), 1)
    exponent = coefficients[0]

    #fit_function_log = lambda x: exponent * x + coefficients[1]

    #fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

    def fit_function_log(x):
        return exponent * x + coefficients[1]

    def fit_function(x):
        return np.exp(fit_function_log(np.log(x)))

    extrapolated_value = fit_function(h_values_ext[-1])

    plt.figure(figsize=(8, 6))
    plt.scatter(h_values_ext, error_values, marker='o',
                color='b', label='Données numériques obtenues')
    plt.plot(h_values_ext, fit_function(h_values_ext), linestyle='--',
             color='r', label='Régression en loi de puissance')

    plt.scatter(h_values_ext[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

    plt.title('Convergence d\'ordre' + Order +
              '\n de l\'erreur '+error_name+' en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)
    # Le paramètre y règle la position verticale du titre

    plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (m)',
               fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
    plt.ylabel('Erreur '+error_name+' (mol/m³) ', fontsize=12, fontweight='bold')

    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

    #equation_text = f'$L_2 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
    equation_text = f'$ {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
    equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12,
                                 transform=plt.gca().transAxes, color='k')

    equation_text_obj.set_position((0.5, 0.4))

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_convergence(errors_values_1_l1, h_values, "1", error_name='L1')
plot_convergence(errors_values_1_l2, h_values, "1", error_name='L2')
plot_convergence(error_linf_1, h_values, "1", error_name='Linf')
plot_convergence(errors_values_2_l2, h_values, "2")
