# import library
import sympy as sp

x, y, eps, nu = sp.symbols('x y eps nu')

u = sp.sin(x*x + y*y) + eps
v = sp.cos(x*x + y*y) + eps
p = sp.sin(x*x + y*y) + 2

s1 = sp.diff(u, x) + sp.diff(v, y)
s2_x = u*sp.diff(u, x) + v* sp.diff(u, y) + sp.diff(p, x) - nu * (sp.diff(u, x, x) + sp.diff(u, y,y))
s2_y = u*sp.diff(v, x) + v* sp.diff(v, y) + sp.diff(p, y) - nu * (sp.diff(v, x, x) + sp.diff(v, y,y))

print('Term source eq. 1:')
print('  - latex:', sp.latex(sp.simplify(s1)))
print('  - c++:',sp.printing.cxxcode(sp.simplify(s1), standard='C++11'))

print()
print('Term source eq. 2, x direction:')
print('  - latex:', sp.latex(sp.simplify(s2_x)))
print('  - c++:',sp.printing.cxxcode(sp.simplify(s2_x), standard='C++11'))

print()
print('Term source eq. 2, y direction:')
print('  - latex:', sp.latex(sp.simplify(s2_y)))
print('  - c++:',sp.printing.cxxcode(sp.simplify(s2_y), standard='C++11'))