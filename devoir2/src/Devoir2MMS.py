# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:04:32 2024

@author: lucas
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def manufactured_solution(r, t):
    Cs = 12.0
    R = 0.5
    return (Cs*np.sin(np.pi*r**2/(2*R**2)) + r**2*(-R + r)/(t + 1))

def source_term(r, t):
    Cs = 12.0
    R = 0.5
    De = 10e-10
    k = 4e-9
    kc = k*(Cs*np.sin(np.pi*r**2/(2*R**2)) + r**2*(-R + r)/(t + 1))
    expression = -De*(np.pi*Cs*np.cos(np.pi*r**2/(2*R**2))/R**2 - np.pi**2*Cs*r**2*np.sin(np.pi*r**2/(2*R**2))/R**4 + 4*r/(t + 1) - 2*(R - r)/(t + 1)) - De*(np.pi*Cs*r*np.cos(np.pi*r**2/(2*R**2))/R**2 + r**2/(t + 1) + 2*r*(-R + r)/(t + 1))/r - r**2*(-R + r)/(t + 1)**2
    return (expression + kc)


def solve_MMS(n, dt, order, imax, tol, time, debug=False):
    '''
    Solves Fick's second law of diffusion using the finite difference method with addition of source term
    
    This function is exactly the same as in "devoir 1", but now we add a source term (-kC) and also the source term
    from MMS (Manufactured Method Solution)

    Args:
        n (int): Number of discretization points.
        dt (float): Time interval.
        order (int): Order of the finite difference method (1 or 2).
        imax (int, optional): Maximum number of iterations. 
        tol (float, optional): Tolerance for the stopping criterion. 
        time(int): current time
        debug (bool, optional): Enable debug mode. Default is False.

    Returns:
        tuple: A tuple containing the simulation results.
    
    '''
    # Geometry
    r0 = 0
    rf = 0.5

    # Constant variables
    d_eff = 10e-10
    # s = 8E-9
    c_e = 12.
    k=4e-9 
    # Position vector
    r = np.linspace(r0, rf, n)
    h = (rf -r0)/(n-1)

    # Creation of the system matrix
    #   bl: diagonal left
    #   a : diagonal
    #   br: diagonal right
    bl_vector = np.zeros(n-2)
    a_vector = np.zeros(n-2)
    br_vector = np.zeros(n-2)

    for i in range(0, n-2):
        if order==1:
            bl_vector[i] = -1*d_eff*dt /(h*h)
            br_vector[i] = -1*d_eff*dt * (1/(r[i+1]*h) + 1/(h*h))
            a_vector[i] = 1 + d_eff*dt * (1/(r[i+1]*h) + 2/(h*h))
        elif order==2:
            bl_vector[i] = -1*d_eff*dt * ( (-1/(2*r[i+1]*h)) + 1/(h*h))
            br_vector[i] = -1*d_eff*dt * ( (1/(2*r[i+1]*h)) + 1/(h*h))
            a_vector[i] = 1 + d_eff*dt * 2/(h*h)

    a = np.diag(a_vector)
    a = np.c_[np.zeros(n-2), a, np.zeros(n-2)]

    bl = np.diag(bl_vector)
    bl = np.c_[bl, np.zeros(n-2), np.zeros(n-2)]

    br = np.diag(br_vector,2)
    br = np.delete(br, [n-2, n-1], 0) #rows

    matrix = bl+a+br

    # Boundary conditions
    # Neuwmann
    bc_r0_vector = np.zeros(n)
    if order == 1:
        bc_r0_vector[0:2] = [1,-1]
    elif order == 2:
        bc_r0_vector[0:3] = [-3, 4, -1]
    # Dirichlet
    bc_rn_vector = np.zeros(n)
    bc_rn_vector[-1] = 1

    matrix = np.r_[ [bc_r0_vector], matrix, [bc_rn_vector]] # Add rows for bc
    matrix_inv = LA.inv(matrix)
 
    c = np.zeros(n)
    c[n-1] =  c_e

    #********** MAIN LOOP **********
    c_pre = np.zeros(n)
    res = 1
    i =0
    while i < imax and abs(res) > tol:
        s_current = source_term(r, time)  #Ici nous calculons le terme source pour r et le time spécifié
        c_pre = c.copy()

        c[1:-1] = c[1:-1] - k*c[1:-1]*dt + s_current[1:-1]*dt


        c = np.linalg.solve(matrix, c)

        res = LA.norm(c - c_pre)

    if (i % 1000 == 0) and debug:
        print(i, res)

    i += 1

    if i == imax:
        print('    ***********')
        print('    Maximal number of iterations achived')

    return r, c

# #Time step
# dt = 1E5
# #Tolérance   
# tol = 1E-15
# #imax
# imax = 1e7
# #nombre de points de grille
# n = 100
# #ordre
# order = 2 
# #Choix du temps
# time = 1e10

#Solver
h=[]
El2=[]
n_cases = [10,20,50,80,100]
for n in n_cases:
    dt = 1E5
    tol = 1e-15
    imax = 1e7
    order = 2
    time = 1e10
    dx=0.5/n
    h.append(dx)
    r1,c_num= solve_MMS(n, dt, order, imax, tol, time)
    c_man = manufactured_solution(r1, time)
    error_l2 = np.sqrt(np.sum((c_num - c_man)**2)/len(c_num))
    El2.append(error_l2)
    print("n :",n)
    print("L2", error_l2)
    plt.figure(figsize=(10, 6))
    plt.plot(r1, c_num, label='Solution Numérique', marker='o')
    plt.plot(r1, c_man, label='Solution Manufacturée', linestyle='--')
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

    #fit_function_log = lambda x: exponent * x + coefficients[1]

    #fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

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
    equation_text = f'$ {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
    equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12,
                                  transform=plt.gca().transAxes, color='k')

    equation_text_obj.set_position((0.5, 0.4))

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
    
El2=[]
dt_cases = [1e3,1e4,1e5,1e6,1e7]
for dt in dt_cases:
    tol = 1e-15
    imax = 1e7
    order = 2
    time = 1e10
    n=200
    r1,c_num= solve_MMS(n, dt, order, imax, tol, time)
    c_man = manufactured_solution(r1, time)
    error_l2 = np.sqrt(np.sum((c_num - c_man)**2)/len(c_num))
    El2.append(error_l2)
    print("n :",n)
    print("L2", error_l2)
    plt.figure(figsize=(10, 6))
    plt.plot(r1, c_num, label='Solution Numérique', marker='o')
    plt.plot(r1, c_man, label='Solution Manufacturée', linestyle='--')
    plt.xlabel('Position (r)')
    plt.ylabel('Concentration (C)')
    plt.title('Comparaison entre la solution numérique et manufacturée')
    plt.legend()
    plt.show()

plot_convergence_t(El2,dt_cases,"2", error_name='L2')