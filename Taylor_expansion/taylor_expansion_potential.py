# Imports
import matplotlib.pyplot as plt
import numpy as np
import math as m

# constants 
λ = 1064*10**(-9)
n = 1.45
w01 = 350*10**(-6)
w02 = 300*10**(-6)
p1 = 150*10**(-3)
p2 = 150*10**(-3)
c = 3*10**8
e0 = 9*10**(-12)
ϵ = 1*10**(-12)
r = 10**(-6)
mass = 1

# define x_0
x0 = 2.0

# Rayleigh range and field amplitude
xR1 = (m.pi * w01**2 * n) / λ
xR2 = (m.pi * w02**2 * n) / λ
P1 = (4 * p1) / (m.pi * w01**2 * c * e0)
P2 = (4 * p2) / (m.pi * w02**2 * c * e0)

# F = (A(x)^2)''+(A(-x)^2)''
# function A is a product of three different functions (named f(x), g(x) and h(x)), for which I calculated the derivatives 
# and then used the derivative of multiplication to tie the whole expression together

f1 = (2 * x0 * xR1**2) / (x0**2 + xR1**2)**2
f2 = (2 * x0 * xR2**2) / (x0**2 + xR2**2)**2

df1 = (2 * (x0**2 - xR1**2) * xR1**2 - 8 * x0**2 * xR1**2) / (x0**2 + xR1**2)**3
df2 = (2 * (x0**2 - xR2**2) * xR2**2 - 8 * x0**2 * xR2**2) / (x0**2 + xR2**2)**3

ddf1 = -(8 * xR1**2 * x0) / (x0**2 + xR1**2)**5 - (16 * x0 * xR1**2 * (x0**2 + xR1**2) - 48 * x0**3 * xR1**2) / (x0**2 + xR1**2)**3
ddf2 = -(8 * xR2**2 * x0) / (x0**2 + xR2**2)**5 - (16 * x0 * xR2**2 * (x0**2 + xR2**2) - 48 * x0**3 * xR2**2) / (x0**2 + xR2**2)**3

g1 = 1 - (2 * r**2 * xR1**2) / (w01**2 * (x0**2 + xR1**2))
g2 = 1 - (2 * r**2 * xR2**2) / (w02**2 * (x0**2 + xR2**2))

dg1 = (8 * r**2 * xR1**2 * x0) / (w01**2 * (x0**2 + xR1**2)**2)
dg2 = (8 * r**2 * xR2**2 * x0) / (w02**2 * (x0**2 + xR2**2)**2)

ddg1 = (8 * r**2 * xR1**2 * (x0**2 + xR1**2) - 32 * r**2 * xR1**2 * x0**2) / (w01**2 * (x0**2 + xR1**2)**3)
ddg2 = (8 * r**2 * xR2**2 * (x0**2 + xR2**2) - 32 * r**2 * xR2**2 * x0**2) / (w02**2 * (x0**2 + xR2**2)**3)

h1 = m.exp(-(2 * r**2 * xR1**2) / (w01**2 * (x0**2 + xR1**2)))
h2 = m.exp(-(2 * r**2 * xR2**2) / (w02**2 * (x0**2 + xR2**2)))

dh1 = dg1 * h1
dh2 = dg2 * h2

ddh1 = ddg1 * h1 + dg1**2 * h1
ddh2 = ddg2 * h2 + dg2**2 * h2

x = np.linspace(x0 - 1, x0 + 1, 500)
xtilde = (x - x0)*10**(-8) # Small interval for particle movement

dA1 = P1 * (ddf1 * g1 * h1 + df1 * dg1 * h1 + df1 * g1 * dh1 + ddg1 * f1 * h1 + dg1 * df1 * h1 + dg1**2 * f1 * h1
                             + ddg1 * h1 * f1 * g1 + dg1**2 * h1 * f1 * g1 + dg1**2 * h1 * f1 + dg1 * h1 * g1 * df1)
dA2 = P2 * (ddf2 * g2 * h2 + df2 * dg2 * h2 + df2 * g2 * dh2 + ddg2 * f2 * h2 + dg2 * df2 * h2 + dg2**2 * f2 * h2
                             + ddg2 * h2 * f2 * g2 + dg2**2 * h2 * f2 * g2 + dg2**2 * h2 * f2 + dg2 * h2 * g2 * df2)

# Ω^2 = 1/m * ∂_x^ V
Ohmega2 = (1 /  mass) * (dA1 + dA2)

V = (1 / 2) * Ohmega2**2 * mass * xtilde**2

plt.plot(xtilde, V)
plt.xlabel(r'$\tilde{x} = x - x_0$')
plt.ylabel('Potential')
plt.title('Potential form the fields Taylor expansions second term')
plt.show()


