import pandas as pd
import numpy as np
import solve_FICK_sourceTerm as mycode
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/lucas/OneDrive/Documents/GitHub/MEC8211/devoir2/data/Comsol_profil_81_1e5.csv', sep=';')
print(df)

plt.plot(df['R'], df['c'])

plt.title('Valeur en fonction de la Date')
plt.xlabel('R')
plt.ylabel('c')
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show()


h=[]
El2=[]
n_cases = [81]
for n in n_cases:
    tMax = 1e12#0.1
    dt = 1e7
    Order = 2
    dx=0.5/n
    h.append(dx)
    r,c_num = mycode.solve(n, dt, Order, tMax, MMS = False, debug=False)
    error_l2 = np.sqrt(np.sum((c_num - df['c'])**2)/len(c_num))
    El2.append(error_l2)
    print("n :",n)
    print("L2", error_l2)
    plt.figure(figsize=(10, 6))
    plt.plot(r, c_num, label='Solution Numérique', marker='o')
    plt.plot(r, df['c'], label='Solution Manufacturée', linestyle='--')
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
