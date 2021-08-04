#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from casadi import *
import numpy as np
from matplotlib import pyplot as plt

# Soft constraint cost for track boundaries
t = 6                               # Threshold
cost_fit = np.zeros((10000))
numbers = np.linspace(-10,10,10000)
for i in range(10000):
    if -t <= numbers[i] <= t:
        cost_fit[i] = 0.0
    else:
        cost_fit[i] = (abs(numbers[i]) - t) ** 2
lut_d = interpolant('LUT_d', 'bspline', [numbers], cost_fit, dict(degree=[3]))

plt.plot(numbers, lut_d(numbers))
plt.axvline(-7.5, color='r')
plt.axvline(-6, color='g', linestyle='--')
plt.axvline(0, color='r', linestyle='--')
plt.axvline(7.5, color='r')
plt.axvline(6, color='g', linestyle='--')
plt.axhline(0, color='gray', linestyle='--')
plt.legend(['Cost Function', 'Track boundaries', 'Safety Margin', 'Track centerline'], fontsize=15)
plt.xlabel('Distance offset from track centerline', fontsize=15)
plt.ylabel('Cost for constraint violation', fontsize=15)
plt.title('Soft constraint for track boundaries', fontsize=15)
plt.show()

x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)

X, Y = np.meshgrid(x, y)

Z = (X / 4.5) ** 2 + (Y / 3) ** 2

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if Z[i, j] >= 1:
            Z[i, j] = 0
        else:
            Z[i, j] = (1 - Z[i, j]) ** 3

plt.plot(0, 0, marker=8, markersize=15)
plt.legend(['Obstacle vehicle center'], fontsize=15)
col = plt.contourf(X, Y, Z, 100, cmap='rainbow')
cbar = plt.colorbar(col)
cbar.set_label('Cost for obstacle collision', fontsize=15)
plt.title('Collision avoidance cost function', fontsize=15)
plt.show()