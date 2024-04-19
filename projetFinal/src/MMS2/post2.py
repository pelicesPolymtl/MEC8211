import vtk
from vtk.util.numpy_support import vtk_to_numpy
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt

alto = 7
ancho = 9

def plot_convergenceP(error_values, h_values_ext, viscosity, order, error_name = 'L2') :
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

    plt.figure(figsize=(ancho, alto))
    plt.scatter(h_values_ext, error_values, marker='o',
                color='b', label='Numerical solution')
    plt.plot(h_values_ext, fit_function(h_values_ext), linestyle='--',
             color='r', label='Power-law regression')

    # plt.scatter(h_values_ext[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

    plt.title('Pressure* convergence as a function of $Δh$. Viscosity='+str(viscosity),
          fontsize=14, fontweight='bold', y=1.02)
    # Le paramètre y règle la position verticale du titre

    plt.xlabel('Mesh size $h_{max}$',
               fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
    plt.ylabel('L2-Error', fontsize=12, fontweight='bold')

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
    plt.savefig('convergenceP.png')


def plot_convergenceU(error_values, h_values_ext, viscosity, order, error_name = 'L2') :
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

    plt.figure(figsize=(ancho, alto))
    plt.scatter(h_values_ext, error_values, marker='o',
                color='b', label='Numerical solution')
    plt.plot(h_values_ext, fit_function(h_values_ext), linestyle='--',
             color='r', label='Power-law regression')

    # plt.scatter(h_values_ext[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

    plt.title('Velocity-X convergence as a function of $Δh$. Viscosity='+str(viscosity),
          fontsize=14, fontweight='bold', y=1.02)
    # Le paramètre y règle la position verticale du titre

    plt.xlabel('Mesh size $h_{max}$',
               fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
    plt.ylabel('L2-Error', fontsize=12, fontweight='bold')

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
    plt.savefig('convergenceU.png')

def plot_convergenceV(error_values, h_values_ext, viscosity, order, error_name = 'L2') :
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

    plt.figure(figsize=(ancho, alto))
    plt.scatter(h_values_ext, error_values, marker='o',
                color='b', label='Numerical solution')
    plt.plot(h_values_ext, fit_function(h_values_ext), linestyle='--',
             color='r', label='Power-law regression')

    # plt.scatter(h_values_ext[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

    plt.title('Velocity-Y convergence as a function of $Δh$. Viscosity='+str(viscosity),
          fontsize=14, fontweight='bold', y=1.02)
    # Le paramètre y règle la position verticale du titre

    plt.xlabel('Mesh size $h_{max}$',
               fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
    plt.ylabel('L2-Error', fontsize=12, fontweight='bold')

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
    plt.savefig('convergenceV.png')



viscosity = 1
h_values =  np.flip(np.array([ 0.05,        0.025,       0.0125,       0.00625,               0.003125]))
errorL2_p = np.flip(np.array([ 0.00085712,  0.000189704, 4.43669e-05,  1.089980132319631e-05, 2.739483792218953e-06]))
errorL2_u = np.flip(np.array([ 2.8276e-05,  4.32561e-06, 6.318e-07,    8.799764398089643e-08, 1.179197641863184e-08]))
errorL2_v = np.flip(np.array([ 1.69338e-05, 2.2618e-06,  2.99526e-07 , 3.910255743161248e-08, 5.033206248118747e-09]))

plot_convergenceP(errorL2_p, h_values, 1, viscosity)
plot_convergenceU(errorL2_u, h_values, 1, viscosity)
plot_convergenceV(errorL2_v, h_values, 1, viscosity)

print(h_values)
print(errorL2_p)
print(errorL2_u)
print(errorL2_v)

print(100*(2.0088 -2)/2)
print(100*(2.8718 -3)/3)
print(100*(2.9475 -3)/3)

