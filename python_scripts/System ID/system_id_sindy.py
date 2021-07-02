#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys
import pickle
from typing import Optional
import numpy as np
from casadi import *
from numpy.lib.utils import info
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.util import *
from Common.custom_dataclass import *

if __name__=='__main__':

    # Read ground truth states
    filename = '../../Data/states_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_s = np.array([d.time for d in data])
    dt = np.average(np.diff(t_s))
    x = np.array([d.pose_x for d in data])
    y = np.array([d.pose_y for d in data])
    yaw = np.array([np.radians(d.pose_yaw) for d in data])
    vx = np.array([d.v_lon for d in data])
    vy = np.array([d.v_lat for d in data])

    # Read control commands
    filename = '../../Data/controls_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_c = np.array([d.time for d in data])
    throttle = np.array([d.throttle for d in data])
    brake = np.array([d.brake for d in data])
    acc = np.array([d.acc for d in data])
    steer = np.array([d.steer * np.pi / 180 for d in data])

    # Create libraray of possible non linearities

    ctheta = np.array([np.cos(theta) for theta in yaw])
    stheta = np.array([np.sin(theta) for theta in yaw])
    vx_ctheta = np.multiply(vx, ctheta)
    vx_stheta = np.multiply(vx, stheta)
    vy_ctheta = np.multiply(vy, ctheta)
    vy_stheta = np.multiply(vy, stheta)

    tdelta = np.array([np.tan(delta) for delta in steer])
    vx_tdelta = np.multiply(vx, tdelta)

    atan_vy_vx = np.arctan2(vy, vx)

    # Nonlinear system 

    # PHI = np.vstack((x[:-1], y[:-1],
    #                  vx_ctheta[:-1], vx_stheta[:-1], vy_ctheta[:-1], vy_stheta[:-1],
    #                  steer[:-1], vx_tdelta[:-1],
    #                  throttle[:-1], brake[:-1],
    #                  atan_vy_vx[:-1]                    
    #                  )).T

    PHI = np.vstack((vx_tdelta[:-1] * dt, throttle[:-1] * dt, brake[:-1] * dt, atan_vy_vx[:-1] - steer[:-1])).T

    np.savetxt('../../Data/PHI.txt', PHI)

    X_DOT = np.vstack((yaw[1:], vx[1:], vy[1:])).T
    np.savetxt('../../Data/X_dot.txt', X_DOT)

    params = np.zeros((X_DOT.shape[1], PHI.shape[1]))

    p = 0
    for dim in range(X_DOT.shape[1]):
        opti = Opti()
        # w = opti.variable(PHI.shape[1], 1)
        w = opti.variable(PHI.shape[1], 1)

        opti.minimize(0.5 * dot(MX(X_DOT[:, dim]) - mtimes(MX(PHI), w), MX(X_DOT[:, dim]) - mtimes(MX(PHI), w)) + p * dot(fabs(w), MX.ones(PHI.shape[1] )))

        opti.solver('ipopt')
        sol = opti.solve()
        params[dim, :] = np.array(sol.value(w))
        print(sol.value(w))

    print(params)

    pred_states = np.zeros((PHI.shape[0], 5))

    pred_states[0, :] = np.array([x[0], y[0], yaw[0], vx[0], vy[0]])

    for i in range(1, pred_states.shape[0]):
        x_, y_, yaw_, vx_, vy_ = pred_states[i - 1, :]
        a = throttle[i - 1]
        b = brake[i - 1]
        delta = steer[i - 1]

        # pred_states[i, 0] = params[0, 0] * x_ + (params[0, 2] * vx_ * np.cos(yaw_) + params[0, 5] * vy_ * np.sin(yaw_)) * dt
        # pred_states[i, 1] = params[1, 1] * y_ + (params[1, 3] * vx_ * np.sin(yaw_) + params[1, 4] * vy_ * np.cos(yaw_)) * dt
        # pred_states[i, 2] = wrapToPi(yaw_ + (params[2, 7] * vx_ * np.tan(delta)) * dt)
        # pred_states[i, 3] = vx_ + (params[3, 8] * a + params[3, 9] * (a ** 2) + params[3, 10] * b + params[3, 11] * (b ** 2)) * dt
        # pred_states[i, 4] = vy_ + (params[4, 12] * np.arctan2(vy_, vx_) - params[4, 6] * delta) * dt

        pred_states[i, 0] = x_ + (vx_ * np.cos(yaw_) + vy_ * np.sin(yaw_)) * dt
        pred_states[i, 1] = y_ + (vx_ * np.sin(yaw_) + vy_ * np.cos(yaw_)) * dt
        pred_states[i, 2] = yaw_ + (params[0, 0] * vx_ * np.tan(delta)) * dt
        pred_states[i, 3] = vx_ + (params[1, 1] * a + params[1, 2] * b) * dt
        pred_states[i, 4] = vy_ + (params[2, 3] * wrapToPi(np.arctan2(vy_, vx_) - delta)) * dt
    
    plt.plot(x, y)
    plt.plot(pred_states[:, 0], pred_states[:, 1])
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('System ID trajectory comparison')
    plt.show()