'''
##################################################
## This module solves Fick's second law
##################################################
## Code to MEC8211
##################################################
## Authors:
##     - Pablo ELICES PAZ
##     - Lucas BRAHIC
##     - Justin BELZILE
## Date: 10/02/2024
##################################################
'''
import numpy as np
from numpy import linalg as LA

def solve(n, dt, order, imax = 100000, tol = 1E-12, debug=False):
    '''
    Solves Fick's second law of diffusion using the finite difference method.

    Args:
        n (int): Number of discretization points.
        dt (float): Time interval.
        order (int): Order of the finite difference method (1 or 2).
        imax (int, optional): Maximum number of iterations. Default is 100000.
        tol (float, optional): Tolerance for the stopping criterion. Default is 1E-12.
        debug (bool, optional): Enable debug mode. Default is False.

    Returns:
        tuple: A tuple containing the simulation results.
    '''
    # Geometry
    r0 = 0.
    rf = 0.5

    # Constant variables
    d_eff = 10E-10
    s = 8E-9
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
    # Diriclet
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

    # Analytical solution
    c_a=0.25*s/d_eff*rf**2.*(r**2/rf**2-1.)+c_e

    # initial condition:
    # C(t=0)=10
    # We set 10 to make it faster
    # C = np.zeros(n)+11
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

        for j in range(n):
            c_pre[j] = c[j]

        c = c-s*dt

        c[0] = 0
        c[n-1] =  c_e

        c = np.matmul(matrix_inv, c)
        res = LA.norm(c-c_pre)

        if (i%1000 == 0) and debug:
            print(i, res)

        i += 1
    if i == imax:
        print('    ***********')
        print('    Maximal number of iterations achived')

    return r, c, c_a
