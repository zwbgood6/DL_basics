# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 20:55:59 2018

@author: Wenbo Zhang
"""
# Z-axis: Gradient w.r.t. weight
# E = e^(WX+b), L = E/(1+E), gradient = 2WE^2/(1+E)^2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

# Make data.
W = np.arange(-2, 2, 0.5)
b = np.arange(-2, 2, 0.5)
W, b = np.meshgrid(W, b)
X = 1
E = np.exp(W*X+b)
Z = 2*W*E**2/(1+E)**2

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(W, b, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.5, 3)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# Set labels
ax.set_xlabel('Weights')
ax.set_ylabel('Bias')
ax.set_zlabel('Gradient')

# title text
ax.text2D(0.35, 0.95, "Gradient of L2 loss", transform=ax.transAxes)

plt.show()