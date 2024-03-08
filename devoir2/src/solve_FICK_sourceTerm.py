# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:04:32 2024

@author: lucas
         pablo
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def manufactured_solution(r,t):
    Cs = 12.0
    R = 0.5
    return (Cs*np.sin(np.pi*r**2/(2*R**2)) - r**2*t**0.25*(-R + r))

def source_term(r, t):
    Cs = 12.0
    R = 0.5
    De = 1e-10
    k = 4e-9
    r = np.maximum(r, 1e-25)
    t = np.maximum(t, 1e-25)
    kc = k*(Cs*np.sin(np.pi*r**2/(2*R**2)) - r**2*t**0.25*(-R + r))

    expression = (
    -De*(np.pi*Cs*np.cos(np.pi*r**2/(2*R**2))/R**2 - np.pi**2*Cs*r**2*np.sin(np.pi*r**2/(2*R**2))/R**4 - 4*r*t**0.25 + 2*t**0.25*(R - r))
    - De*(np.pi*Cs*r*np.cos(np.pi*r**2/(2*R**2))/R**2 - r**2*t**0.25 - 2*r*t**0.25*(-R + r))/r
    - 0.25*r**2*(-R + r)/t**0.75
    )
    #expression = -De*(np.pi*Cs*np.cos(np.pi*r**2/(2*R**2))/R**2 - np.pi**2*Cs*r**2*np.sin(np.pi*r**2/(2*R**2))/R**4 -
    # 4*r*t**0.25 + 2*t**0.25*(R - r)) - De*(np.pi*Cs*r*np.cos(np.pi*r**2/(2*R**2))/R**2 -
    # r**2*t**0.25 - 2*r*t**0.25*(-R + r))/r - 0.25*r**2*(-R + r)/t**0.75
    return expression + kc

def solve(n, dt, order, tmax,  MMS = True, debug=False):
    '''
    Solves Fick's second law of diffusion using the finite difference method 
    with addition of source term
    
    This function is exactly the same as in "devoir 1", but now we add a source 
    term (-kC) and also the source term from MMS (Manufactured Method Solution)

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
    k = 4e-9

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
    c[n-1] = c_e

    #********** MAIN LOOP **********
    c_pre = np.zeros(n)
    i =0; t=0
    while t < tmax:
        c_pre = c.copy()

        if MMS:
            s_current = source_term(r, t)
            c = c - k*c*dt + s_current*dt #Ici nous calculons le terme source pour r et le time spécifié
        else: c = c - k*c*dt 

        c[0]=0.;c[-1]=12
        c = np.matmul(matrix_inv, c)
        #c = LA.solve(matrix, c)
        res = LA.norm(c - c_pre)

        if i==30:
            print(c[0])

        i += 1; t += dt

    print('number of iteration: ',i)
    return r, c
