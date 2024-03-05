import numpy as np
import solve_FICK_sourceTerm as mycode
import matplotlib.pyplot as plt

n = 20
tFinal = 15
for t in np.linspace(0,tFinal,n+1):
    # Geometry
    r0 = 0
    rf = 0.5
    r = np.linspace(r0, rf, n)
    c_an = mycode.manufactured_solution(r, t)
    plt.plot(r, c_an, label='time t='+str(t))
plt.legend()
plt.show()
