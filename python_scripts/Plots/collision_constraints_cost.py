#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from casadi import *
import numpy as np
import imageio
import shutil, os
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

f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('Soft constraint for track boundaries', fontsize=20)
# ax1.set_aspect('equal', 'box')
# ax2.set_aspect('equal', 'box')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

ax1.plot(numbers, lut_d(numbers))
ax1.axvline(-7.5, color='r')
ax1.axvline(-6, color='g', linestyle='--')
ax1.axvline(0, color='r', linestyle='--')
ax1.axvline(7.5, color='r')
ax1.axvline(6, color='g', linestyle='--')
ax1.axhline(0, color='gray', linestyle='--')
ax1.legend(['Cost Function', 'Track boundaries', 'Safety Margin', 'Track centerline'], fontsize=15)
ax1.set_xlabel('Distance offset from track centerline [m]', fontsize=18)
ax1.set_ylabel('Cost for boundary constraint violation', fontsize=18)

x = np.linspace(-10, 10, 1000)
y = np.linspace(0, 10, 1000)

X, Y = np.meshgrid(x, y)

X = X.reshape(-1)
Z = np.zeros_like(X)
for i in range(len(Z)):
    if -t <= X[i] <= t:
        Z[i] = 0
    else:
        Z[i] = (abs(X[i]) - t) ** 2

X = X.reshape(Y.shape)
Z = Z.reshape(Y.shape)

ax2.axvline(-7.5, color='black')
ax2.axvline(-6, color='g', linestyle='--')
ax2.axvline(0, color='r', linestyle='--')
ax2.axvline(7.5, color='black')
ax2.axvline(6, color='g', linestyle='--')
ax2.legend(['Track boundaries', 'Safety Margin', 'Track centerline'], fontsize=15)
col = ax2.contourf(X, Y, Z, 100, cmap='rainbow')
cbar = plt.colorbar(col)
cbar.set_label('Cost for boundary constraint violation', fontsize=15)
ax2.set_xlabel('Distance offset from track centerline [m]', fontsize=18)
ax2.set_ylabel('Along Track distance [m]', fontsize=18)
plt.show()

try:
    os.mkdir('temp/')
except FileExistsError:
    shutil.rmtree('temp/')
    os.mkdir('temp/')

x = np.linspace(-8.5, 8.5, 1000)
y = np.linspace(-10, 10, 1000)

X, Y = np.meshgrid(x, y)

Z1 = ((X - 1) / 3) ** 2 + ((Y + 3) / 4.5) ** 2
Z2 = ((X + 1) / 3) ** 2 + ((Y - 3) / 4.5) ** 2

Z = np.minimum(Z1, Z2)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if Z[i, j] >= 1:
            Z[i, j] = 0
        else:
            Z[i, j] = (1 - Z[i, j]) ** 3

car_y = np.linspace(-5, 5, 100)

f, ax = plt.subplots()
# plt.tight_layout()
ax.set_aspect('equal', 'box')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
for i in range(len(car_y)):
    ax.axvline(-7.5, color='r', label='Track Boundaries')
    ax.axvline(7.5, color='r')
    col = plt.contourf(X, Y, Z, 100, cmap='rainbow')
    cbar = plt.colorbar(col, location='bottom', shrink=0.35)
    cbar.set_label('Cost for obstacle collision', fontsize=15)
    ax.scatter(1, -3, s=100, marker='x', color='black', label='Obstacles')
    ax.scatter(-1, 3, s=100, marker='x', color='black')
    ax.scatter(0, car_y[i], s=100, marker='x', color='green', label='Vehicle')
    cost = min(((0 - 1) / 3) ** 2 + ((car_y[i] + 3) / 4.5) ** 2, ((0 + 1) / 3) ** 2 + ((car_y[i] - 3) / 4.5) ** 2)
    cost = (1 - cost) ** 3
    plt.text(0.5, 0.95, 'Collision Cost: {}'.format(round(cost, 4)), transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=15)
    plt.legend(fontsize=15)
    ax.set_xlabel('Distance offset from track centerline [m]', fontsize=18)
    ax.set_ylabel('Along Track distance [m]', fontsize=18)   
    plt.suptitle('Collision avoidance cost function', fontsize=18)
    plt.pause(0.1)
    plt.savefig('temp/{}.png'.format(i+1))
    ax.clear()
    cbar.remove()

# Generate GIF
with imageio.get_writer('../../GIFS/Collision_avoidance.gif', mode='I') as writer:
    for filename in ['temp/{}.png'.format(i+1) for i in range(len(car_y))]:
        image = imageio.imread(filename)
        writer.append_data(image)
shutil.rmtree('temp/')