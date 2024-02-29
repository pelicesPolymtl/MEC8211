'''
##################################################
## MMS 
##################################################
## Code to MEC8211
##################################################
## Authors:
##     - Lucas BRAHIC
##     - Pablo ELICES PAZ
##     - Justin BELZILE
## Date: 29/02/2024
##################################################
## ToDo:
##     -
##################################################
'''

import numpy as np
from numpy import linalg as LA
from math import pi, exp
import matplotlib.pyplot as plt



def manufactured_solution(r, t, R=0.5, Ce=12):
    return np.exp(-t) * (r**2 - R**2) + Ce * np.sin((np.pi/2) * (r/R))




def source_term(r, t):
        R = 0.5
        Ce = 12
        Deff = 10E-10
        
        #Vérification que r ou t ne soient pas égaux à 0
        #Si oui on remplace la valeur nulle par 1e-25
        r = np.where(r < 1e-25, 1e-25, r)        
        t = np.where(t < 1e-25, 1e-25, t)
        
        
        return (-pi * Ce * Deff * np.cos(pi * r / (2 * R)) / (2 * R * r) +
                pi**2 * Ce * Deff * np.sin(pi * r / (2 * R)) / (4 * R**2) -
                4 * Deff * exp(-t) + R**2 * exp(-t) - r**2 * exp(-t))




def solve_MMS(n, dt, order, imax, tol, time, debug=False):
    '''
    Solves Fick's second law of diffusion using the finite difference method with addition of source term

    Args:
        n (int): Number of discretization points.
        dt (float): Time interval.
        order (int): Order of the finite difference method (1 or 2).
        imax (int, optional): Maximum number of iterations. Default is 100000.
        tol (float, optional): Tolerance for the stopping criterion. Default is 1E-12.
        time(int): current time
        debug (bool, optional): Enable debug mode. Default is False.

    Returns:
        tuple: A tuple containing the simulation results.
    '''
    # Geometry
    r0 = 0
    rf = 0.5

    # Constant variables
    d_eff = 10E-10
    # s = 8E-9
    c_e = 12.

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

    if debug:
        print('** system matrix ** ')
        print('diagonal vectors: ')
        print(bl_vector, a_vector,  br_vector)
        print('matrix: ')
        print(matrix)

   
    c = np.zeros(n)
    # Dircihlet bc
    c[n-1] =  c_e

    #********** MAIN LOOP **********
    c_pre = np.zeros(n)
    res = 1
    i =0
    if debug:
        print('** main loop **')
        print('max number of iteration : ', imax)
        
    
    while i <imax and abs(res) > tol:
        
        
        if debug:
            print(i)
            
        #Calcul du terme source
    
        s_current = source_term(r, time)
        
        rhs = c.copy()
        for j in range(1, n-2):
            rhs[j] += s_current[j] * dt
        rhs[0] = 0  
        
        c = np.matmul(matrix_inv, rhs)
        res = LA.norm(c-c_pre)

        if (i%1000 == 0) and debug:
            print(i, res)

        i += 1
    if i == imax:
        print('    ***********')
        print('    Maximal number of iterations achived')

    return r, c

#Time step
dt = 1E5
#Tolérance   
tol = 1E-15
#imax
imax = 1E4
#nombre de points de grille
n = 1000
#ordre
order = 2  

#Choix du temps
time = 1000000

#Solver
r1,c_num= solve_MMS(n, dt, order, imax, tol, time)
c_man = manufactured_solution(r1, time)


plt.figure(figsize=(10, 6))
plt.plot(r1, c_num, label='Solution Numérique', marker='o')
plt.plot(r1, c_man, label='Solution Manufacturée', linestyle='--')
plt.xlabel('Position (r)')
plt.ylabel('Concentration (C)')
plt.title('Comparaison entre la solution numérique et manufacturée')
plt.legend()
plt.show()
