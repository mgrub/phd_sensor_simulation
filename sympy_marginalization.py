import sympy.matrices as sm
from sympy import Symbol, symbols, simplify, collect, factor

a = Symbol("a")
b = Symbol("b")
sy = Symbol("sigma_y")

N = 3
Xos = symbols(" ".join([f"Xo{i}" for i in range(N)]))
Xas = symbols(" ".join([f"Xa{i}" for i in range(N)]))
Ys = symbols(" ".join([f"Y{i}" for i in range(N)]))
sigma_xs = symbols(" ".join([f"sigma_x{i}" for i in range(N)]))

Ux = sm.diag(*sigma_xs)
Xo = sm.Matrix(Xos)
Y = sm.Matrix(Ys)


# required matrices
F1 = a**2 / sy**2 * sm.eye(N)
F2 = a / sy**2 * (b * sm.ones(N,1) - Y)
V_inv = F1 + Ux**(-1)
V = V_inv**(-1)
M = V*(Ux**(-1)*Xo - F2)

W_inv = F1 + V_inv
W = W_inv**(-1)
S = W*(-F2 + V_inv*M)

expr = M.T*V_inv*M - S.T*W_inv*S

collected_expr = collect(expr[0,0], a)

for e in expr[0,0].args:
    print(e)