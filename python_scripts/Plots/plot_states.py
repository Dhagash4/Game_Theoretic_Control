#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import pickle
import sys
import numpy as np
from casadi import *
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.util import *
from Common.custom_dataclass import *

# Desired Trajectory
filename = '../../Data/2D_waypoints.txt'
data = np.loadtxt(filename)
x_des = data[:, 0]
y_des = data[:, 1]

# Read ground truth states
filename = 'states_mpc.pickle'
with open(filename, 'rb') as f:
    data = pickle.load(f)

t_states = np.array([d.time for d in data])
x_gt = np.array([d.pose_x for d in data])
y_gt = np.array([d.pose_y for d in data])
yaw_gt = np.array([d.pose_yaw for d in data])
vx_gt = np.array([d.v_lon for d in data])

plt.plot(np.arange(0, len(vx_gt)), vx_gt)
plt.title('Velocity Plot')
plt.xlabel('Control Steps')
plt.ylabel('Longitudinal velocity [m/s]')
plt.show()

fig = plt.figure("Ground Truth v/s Reference Trajectory Comparison")
fig.suptitle("Ground Truth v/s Reference Trajectory Comparison", size=25)
plt.subplot(2, 2, 1)
plt.plot(x_gt, y_gt)
plt.plot(x_des, y_des)
plt.legend(['ground truth', 'center-line'], fontsize=15)
plt.subplot(2, 2, 2)
plt.plot(x_gt, y_gt)
plt.plot(x_des, y_des)
plt.legend(['ground truth', 'center-line'], fontsize=15)
plt.subplot(2, 2, 3)
plt.plot(x_gt, y_gt)
plt.plot(x_des, y_des)

plt.legend(['ground truth', 'center-line'], fontsize=15)
plt.subplot(2, 2, 4)
plt.plot(x_gt, y_gt)
plt.plot(x_des, y_des)
plt.legend(['ground truth', 'center-line'], fontsize=15)
plt.show()
