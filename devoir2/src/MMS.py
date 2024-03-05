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


#Etude de la convergence en espace 

Order = 2

print('runing solver...')
print(' order=', Order)
print('')

h=[]
El2=[]
n_cases = [10,20,40,50,60]
for n in n_cases:
    tMax = 1e15#0.1
    dt = tMax/200
    Order = 2
    dx=0.5/n
    h.append(dx)
    r,c_num = mycode.solve(n, dt, Order, tMax, MMS = True, debug=False)
    c_man = mycode.manufactured_solution(r, tMax)
    error_l2 = np.sqrt(np.sum((c_num - c_man)**2)/len(c_num))
    El2.append(error_l2)
    print("n :",n)
    print("L2", error_l2)
    plt.figure(figsize=(10, 6))
    plt.plot(r, c_num, label='Solution Numérique', marker='o')
    plt.plot(r, c_man, label='Solution Manufacturée', linestyle='--')
    plt.xlabel('Position (r)')
    plt.ylabel('Concentration (C)')
    plt.title('Comparaison entre la solution numérique et manufacturée')
    plt.legend()
    plt.show()


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

plot_convergence(El2, h, "2", error_name='L2')

n=100

print('time t=', tMax)
r,c_num = mycode.solve(n, dt, Order, tMax, MMS = True, debug=False)
c_man = mycode.manufactured_solution(r, tMax)
error_l2 = np.sqrt(np.sum((c_num - c_man)**2)/len(c_num))
print("error_L2 :", error_l2)
plt.plot(r, c_num, label='numeric')
plt.plot(r, c_man, label='analytical')
plt.legend()
plt.show()



#Etude de la connvergence en temps  



def plot_convergence_t(error_values, delta_t, Order, error_name = 'L2') :
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

    coefficients = np.polyfit(np.log(delta_t[:3]), np.log(error_values[:3]), 1)
    exponent = coefficients[0]


    def fit_function_log(x):
        return exponent * x + coefficients[1]

    def fit_function(x):
        return np.exp(fit_function_log(np.log(x)))

    extrapolated_value = fit_function(delta_t[-1])

    plt.figure(figsize=(8, 6))
    plt.scatter(delta_t, error_values, marker='o',
                color='b', label='Données numériques obtenues')
    plt.plot(delta_t, fit_function(delta_t), linestyle='--',
              color='r', label='Régression en loi de puissance')

    plt.scatter(delta_t[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

    plt.title('Convergence d\'ordre' + Order +
              '\n de l\'erreur '+error_name+' en fonction de $Δt$',
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
    equation_text = f'$ {np.exp(coefficients[1]):.1e} \\times \\Delta x^{{{exponent:.4f}}}$'
    equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12,
                                 transform=plt.gca().transAxes, color='k')

    equation_text_obj.set_position((0.5, 0.4))

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
    
El2=[]
time_l = [50,75,85,100,200]   #nombre d'itérations
time = []
for t in time_l:
    n=100
    tMax = 1e15 #0.1
    dt = tMax/t
    Order = 2
    time.append(dt)
    r,c_num = mycode.solve(n, dt, Order, tMax, MMS = True, debug=False)
    c_man = mycode.manufactured_solution(r, tMax)
    error_l2 = np.sqrt(np.sum((c_num - c_man)**2)/len(c_num))
    El2.append(error_l2)
    print("n :",n)
    print("L2", error_l2)
    plt.figure(figsize=(10, 6))
    plt.plot(r, c_num, label='Solution Numérique', marker='o')
    plt.plot(r, c_man, label='Solution Manufacturée', linestyle='--')
    plt.xlabel('Position (r)')
    plt.ylabel('Concentration (C)')
    plt.title('Comparaison entre la solution numérique et manufacturée')
    plt.legend()
    plt.show()

plot_convergence_t(El2,time,"2", error_name='L2')

