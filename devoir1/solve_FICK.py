##################################################
## Solve Fick second law
##################################################
## Code to MEC8211
##################################################
## Authors: 
##     - Pablo ELICES PAZ
##     - 
##     - 
## Date: 06/02/2024
##################################################

import numpy as np
from numpy import linalg as LA 



def solve(N, dt, order, imax = 100000, tol = 1E-12, debug=False):
    # Geometry
    R0 = 0.
    Rf = 0.5

    # Constant variables
    Deff = 10E-10
    S = 8E-9
    Ce = 12.


    # Position vector
    r = np.linspace(R0, Rf, N)
    h = (Rf -R0)/(N-1)

    # Creation of the system matrix
    #   Bl: diagonal left
    #   A : diagonal
    #   Br: diagonal right
    Bl_vector = np.zeros(N-2)
    A_vector = np.zeros(N-2)
    Br_vector = np.zeros(N-2)

    for i in range(0, N-2):
        if order==1:
            Bl_vector[i] = -1*Deff*dt /(h*h)
            Br_vector[i] = -1*Deff*dt * (1/(r[i+1]*h) + 1/(h*h))
            A_vector[i] = 1 + Deff*dt * (1/(r[i+1]*h) + 2/(h*h))
        elif order==2:
            Bl_vector[i] = -1*Deff*dt * ( (-1/(2*r[i+1]*h)) + 1/(h*h))
            Br_vector[i] = -1*Deff*dt * ( (1/(2*r[i+1]*h)) + 1/(h*h))
            A_vector[i] = 1 + Deff*dt * 2/(h*h)

    A = np.diag(A_vector)
    A = np.c_[np.zeros(N-2), A, np.zeros(N-2)]

    Bl = np.diag(Bl_vector)
    Bl = np.c_[Bl, np.zeros(N-2), np.zeros(N-2)]

    Br = np.diag(Br_vector,2)
    Br = np.delete(Br, [N-2, N-1], 0) #rows

    matrix = Bl+A+Br

    # Boundary conditions
    # Neuwmann
    BC_r0_vector = np.zeros(N)
    if order == 1: BC_r0_vector[0:2] = [1,-1]
    elif order == 2: BC_r0_vector[0:3] = [-3, 4, -1]
    # Diriclet
    BC_rN_vector = np.zeros(N)
    BC_rN_vector[-1] = 1


    matrix = np.r_[ [BC_r0_vector], matrix, [BC_rN_vector]] # Add rows for BC
    matrixInv = LA.inv(matrix)

    if debug: 
       print('** system matrix ** ')
       print('diagonal vectors: ')
       print(Bl_vector, A_vector,  Br_vector)
       print('matrix: ')
       print(matrix)


    # Analytical solution
    C_a=1./4.*S/Deff*Rf**2.*(r**2/Rf**2-1.)+Ce


    # initial condition: 
    # C(t=0)=10
    # We set 10 to make it faster
    # C = np.zeros(N)+11 
    C = np.zeros(N)
    # Dircihlet BC
    C[N-1] =  Ce 


    #********** MAIN LOOP **********
    C_pre = np.zeros(N)
    res = 1; i =0
    if debug:
       print('** main loop **')
       print('max number of iteration : ', imax)

    while i <imax and abs(res) > tol:
       
       for j in range(N): C_pre[j] = C[j]
       
       C = C-S*dt

       C[0] = 0
       C[N-1] =  Ce

       C = np.matmul(matrixInv, C)
       res = LA.norm(C-C_pre)
       
       if (i%1000 == 0) and debug: print(i, res)

       i += 1
    if i == imax: 
        print('    ***********')
        print('    Maximal number of iterations achived')

    return r, C, C_a