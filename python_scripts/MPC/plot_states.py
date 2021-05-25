import glob
import getopt
import pickle
import os, sys
import numpy as np
from casadi import *
from numpy.lib.utils import info
from scipy.sparse import csc_matrix
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
filename = 'states.pickle'
with open(filename, 'rb') as f:
    data = pickle.load(f)

t_states = np.array([d.time for d in data])
x_gt = np.array([d.pose_x for d in data])
y_gt = np.array([d.pose_y for d in data])
yaw_gt = np.array([d.pose_yaw for d in data])


# # Read control commands
# filename = '../Data/controls_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
# with open(filename, 'rb') as f:
#     data = pickle.load(f)

# t_controls = np.array([d.time for d in data])
# throttle = np.array([d.throttle for d in data])
# brake = np.array([d.brake for d in data])
# steer = np.array([d.steer for d in data])

plt.plot(x_gt, y_gt)
plt.plot(x_des, y_des)
plt.plot(x_gt[0], y_gt[0], 'xr')
plt.plot(x_gt[-1], y_gt[-1], 'xg')
plt.show()