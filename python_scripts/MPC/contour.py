import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpolate
from scipy.misc import derivative
from scipy import interpolate as interp
from casadi import *

waypoints = np.loadtxt('../../Data/2D_waypoints.txt')

L = np.arange(0, waypoints.shape[0])

lut_x = interpolant('LUT_x', 'bspline', [L], waypoints[:, 0], dict(degree=[3]))
lut_y = interpolant('LUT_y', 'bspline', [L], waypoints[:, 1], dict(degree=[3]))
lut_theta = interpolant('LUT_t', 'bspline', [L], waypoints[:, 2], dict(degree=[1]))

pred_x = lut_x(np.arange(0, waypoints.shape[0]-1, 0.01))
pred_y = lut_y(np.arange(0, waypoints.shape[0]-1, 0.01))
pred_theta = lut_theta(np.arange(0, waypoints.shape[0]-1, 0.01))

plt.plot(L, waypoints[:, 2], label='true')
plt.plot(np.arange(0, waypoints.shape[0]-1, 0.01), pred_theta, label='predict')
plt.legend()
plt.show()
plt.plot(waypoints[:, 1], waypoints[:, 0], label='true')
plt.plot(pred_y, pred_x, label='predicted')
plt.legend()
plt.show()

