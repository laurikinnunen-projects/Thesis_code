# Imports
import numpy as np
import matplotlib.pyplot as plt

# Simplified dual beam setup for gaussian beams

# Parameters
w = 1.0        # beam waist
k = 2*np.pi    # wave number (lambda = 1)
I0 = 1.0       # peak intensity

# range
x = np.linspace(-2, 2, 400) # Axial coordinate 
z = np.linspace(-2, 2, 400) # Radial coordinate (y=0)
X, Z = np.meshgrid(x, z)

# E_tot ​= E_plus ​+ E_minus ​=exp(−X2/w2)(eikZ+e−ikZ)= 2exp(−X2/w2)cos(kZ)
# I = |E_tot|^2
I = I0 * np.exp(-2*Z**2 / w**2) * np.cos(k*X)**2

# Plot
plt.figure(figsize=(7,4))
plt.plot([x.min(),x.max()],[0,0], 'r', lw = 2)
plt.imshow(I, extent=[x.min(), x.max(), z.min(), z.max()],
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel('z (radial)')
plt.ylabel('x (axial)')
plt.title('Counter propagating gaussian beam intensity profile')
plt.show()
