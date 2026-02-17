# Imports
import matplotlib.pyplot as plt
import numpy as np
import math as m
import numdifftools as nd

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
r = 200*10**(-6)
mass = 1
pol = 1
k = (2 * np.pi) / λ

# Rayleigh range and field amplitude
xR1 = (m.pi * w01**2 * n) / λ
xR2 = (m.pi * w02**2 * n) / λ
P1 = (4 * p1) / (m.pi * w01**2 * c * e0)
P2 = (4 * p2) / (m.pi * w02**2 * c * e0)


# Function of the real field for the numerical solver
def V(x_var):
    return -0.25 * pol * (P1 * (1 + (x_var / xR1)**2)**(-1) * np.exp(-2 * r**2 / (w01**2 * (1 + (x_var / xR1)**2))) 
                         + P2 * (1 + (x_var / xR2)**2)**(-1) * np.exp(-2 * r**2 / (w02**2 * (1 + (x_var / xR2)**2))))

# x_0 from the roots of the potential first derivative
# Root is r < w0
x0 = 0

# Roots if r > w0
#pos_x01 = xR1 * np.sqrt((2 * r**2) / w01**2 - 1)
#pos_x02 = xR2 * np.sqrt((2 * r**2) / w02**2 -1)

#neg_x01 = -xR1 * np.sqrt((2 * r**2) / w01**2 - 1)
#neg_x02 = -xR2 * np.sqrt((2 * r**2) / w02**2 - 1)

# Interval for potential on x-axis around the minima
x = np.linspace(x0 - 1, x0 + 1, 500)
xtilde = (x - x0)

# V = -0.5 * alpha * |E_tot(r,x)|^2 = -0.5 * alpha * (E(r, x)^2 + E(r, -x)^2)
# Harmonic potential 
# Second order derivative at x0 = 0, calculated analytically
d1 = 2 / (xR1**2) * (2 * r**2 / w01**2 - 1) * np.exp(- (2 * r**2) / w01**2)
d2 = 2 / (xR2**2) * (2 * r**2 / w02**2 - 1) * np.exp(- (2 * r**2) / w02**2)

# Squared frequency Ω² = (1/m)∂²V
ohmega2 = - 1 / (4 * mass) * pol * (P1 * d1 + P2 * d2)

# Analytically calculated harmonic potential V = (1/2)mΩ²x² plus constant term from taylor epansion
V_harm = 0.5 * mass * ohmega2 * xtilde**2 + V(x0)

# Real field exponential factors
exp_factor1 = np.exp(-1j * (k * xtilde + (k * r**2)/(2 * xtilde * (1 + (xR1 / xtilde)**2)) - np.arctan(xtilde/xR1)))
exp_factor2 = np.exp(-1j * (- k * xtilde + (k * r**2)/(- 2 * xtilde * (1 + (xR2 / xtilde)**2)) - np.arctan(-xtilde/xR2)))

# Anharmonic potential analytical form
V_nharm = -0.25 * pol * (P1 * (1 + (xtilde / xR1)**2)**(-1) * np.exp(-2 * r**2 / (w01**2 * (1 + (xtilde / xR1)**2))) 
                         + P2 * (1 + (xtilde / xR2)**2)**(-1) * np.exp(-2 * r**2 / (w02**2 * (1 + (xtilde / xR2)**2))))

# Numerical solution for reference
d2f = nd.Derivative(V, n = 2)
d2f_at_x0 = d2f(x0)
V_harm_num = 0.5 * d2f_at_x0 * xtilde**2 + V(x0)

# Plots
plt.plot(xtilde, V_nharm, label = 'V anharm')
plt.plot(xtilde, V_harm, label = 'V harm')
plt.plot(xtilde, V_harm_num, 'b--', label = 'V harm num', linewidth = 0.5)
plt.legend()
plt.xlabel(r'$\tilde{x} = x - x_0$')
plt.ylabel('Potential')
plt.show()
print(min(V_harm))


