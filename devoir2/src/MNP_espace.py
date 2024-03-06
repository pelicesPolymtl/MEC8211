# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:43:57 2024

@author: lucas
"""

import numpy as np
from numpy import linalg as LA
import solve_FICK_sourceTerm as mycode
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


R=0.5
n=160
dt=1e7
tMax=1e12
Order=2
Deff = 1e-10
k = 4e-9
R=0.5
dt=1e7
tMax=1e12
Order=2
r,u_MNP_analytique =  mycode.solve(n, dt, Order, tMax,MMS=False, debug=False)
spline = CubicSpline(r, u_MNP_analytique)

def terme_source_MNP(r,spline):
    C = spline(r)
    grad_r = np.gradient(C, r)
    grad_r_2 = np.gradient(grad_r, r)
    Deff = 1e-10
    k = 4e-9
    r = np.maximum(r, 1e-25)
    s = (-Deff/r)*grad_r-Deff*grad_r_2+k*C
    return (s)


#Cette fonction solve est exactement la même que dans solve_Fick_sourTerm mais pour des soucis de simplicité
#Nous crééons une nouvelle fonction solve_MNP, la résolution est la même

def solve_MNP(n, dt, order, tmax, debug=False):
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
    d_eff = 1e-10
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
    i =0; t=0
    while t < tmax:

        c_pre = c.copy()

    
        s_MNP = terme_source_MNP(r,spline)
        c = c - k*c*dt + s_MNP*dt

        c[0]=0.;c[-1]=12 
        c = np.matmul(matrix_inv, c)
        
        #c = LA.solve(matrix, c)

        res = LA.norm(c - c_pre)

        if i==30:
            print(c[0])
        
        i += 1; t += dt

    print('number of iteration: ',i)
    return r, c


h=[]
n_cases = [20,40,80,160]
El2 = []

for n in (n_cases):
    h.append(0.5/n)
    r,c_num = solve_MNP(n, dt, Order, tMax, debug=False)
    u_analytique = spline(r)
    error_l2 = np.sqrt(np.sum((c_num - u_analytique)**2)/len(c_num))
    El2.append(error_l2)
    print("n :",n)
    print("L2", error_l2)
    plt.figure(figsize=(10, 6))
    plt.plot(r, c_num, label='Solution Numérique', marker='o')
    plt.plot(r, u_analytique, label='Solution Analytique', linestyle='--')
    plt.xlabel('Position (r)')
    plt.ylabel('Concentration (C)')
    plt.title('Comparaison entre la solution numérique et Analytique MNP')
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
