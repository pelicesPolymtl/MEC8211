'''
##################################################
## Convergence numerical solution
##################################################
## Code to MEC8211
##################################################
## Authors:
##     - Pablo ELICES PAZ
##     - Lucas BRAHIC
##     - Justin BELZILE
## Date: 07/03/2024
##################################################
## ToDo:
##     -
##################################################
'''

import numpy as np
import matplotlib.pyplot as plt
import solve_FICK_sourceTerm as mycode

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

    # Fit a power law curve to the error data
    coefficients = np.polyfit(np.log(h_values_ext[:3]), np.log(error_values[:3]), 1)
    exponent = coefficients[0]

    # Define functions for the power law curve and its logarithm
    def fit_function_log(x):
        return exponent * x + coefficients[1]

    def fit_function(x):
        return np.exp(fit_function_log(np.log(x)))

     # Extrapolate the error for the maximum grid size
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

# Etude de la convergence en espace
Order = 2

print('runing solver...')
print(' order=', Order)
print('')

h=[]
El2=[]
tMax = 1e12
dt = 1E7
n_cases = [80, 160, 320, 640, 1280]
Order = 2
for i in range(len(n_cases)):
    n = n_cases[i]

    r,c_num = mycode.solve(n+1, dt, Order, tMax, MMS = False, debug=False)
    index = np.linspace(0,n,int(n/2**i)+1, dtype=int)
    if i>0:
        error_l2 = np.sqrt(np.sum((c_num[index] - c_pre)**2)/len(c_num[index]))
        El2.append(error_l2)
        h.append(0.5/n)
        print(n, El2)
    c_pre  = c_num[index]
    print()

print(h)
print(El2)
plot_convergence(El2, h, "2", error_name='L2')
plt.savefig("../results/convergence_o2_numerique.png")

h=[]
El2=[]
tMax = 1e12
dt = 1E7
n_cases = [80,160,320, 640, 1280]
Order = 1

for i in range(len(n_cases)):
    n = n_cases[i]

    r,c_num = mycode.solve(n+1, dt, Order, tMax, MMS = False, debug=False)
    index = np.linspace(0,n,int(n/2**i)+1, dtype=int)
    if i>1:
        error_l2 = np.sqrt(np.sum((c_num[index] - c_pre)**2)/len(c_num[index]))
        El2.append(error_l2)
        h.append(0.5/n)
    c_pre  = c_num[index]
    print()

print(h)
print(El2)
plot_convergence(El2, h, "1", error_name='L2')
plt.savefig("../results/convergence_o1_numerique.png")
