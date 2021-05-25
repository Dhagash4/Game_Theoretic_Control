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


def Jacobian(states, control, params):

    x, y, theta, vx, vy = states
    v = np.sqrt(vx ** 2 + vy ** 2)
    acc, delta = control
    L, p, Cd, Cfr, Ccs = params    
    
    dtheta_dL = (-v * np.tan(delta) * np.cos(np.arctan2(np.tan(delta), 2))) / (L ** 2)

    dvx_dp = acc
    dvx_dCd = -1 * v * vx
    dvx_dCfr = -1 * vx

    dvy_dCd = -1 * v * vy
    dvy_dCfr = -1 * vy
    dvy_dCcs = -wrapToPi(np.arctan2(vy, vx) - delta)

    J = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [dtheta_dL, 0, 0, 0, 0],
                  [0, dvx_dp, dvx_dCd, dvx_dCfr, 0],
                  [0, 0, dvy_dCd, dvy_dCfr, dvy_dCcs]])

    return J


def predict_new(old_states, control, params, dt):
    L, p, Cd, Cfr, Ccs = params
    new_states = np.zeros_like(old_states)
    m, n = old_states.shape
    for i in range(m):
        x, y, theta, vx, vy = old_states[i]
        v = np.sqrt(vx ** 2 + vy ** 2)
        acc, delta = control[i]
        x_new = x + (v * np.cos(np.arctan2(np.tan(delta), 2) + theta) * dt)
        y_new = y + (v * np.sin(np.arctan2(np.tan(delta), 2) + theta) * dt)
        theta_new = wrapToPi(theta + (v * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
        vx_new = vx + (p * acc - Cd * v * vx - Cfr * vx) * dt
        vy_new = vy - (Ccs * wrapToPi(np.arctan2(vy, vx) - delta) + (Cd * v + Cfr) * vy) * dt
        new_states[i] = [x_new, y_new, theta_new, vx_new, vy_new]

    return new_states


if __name__=='__main__':

    # Read ground truth states
    filename = '../../Data/states_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_s = np.array([d.time for d in data])
    t_s = t_s[:-1]
    dt = np.average(np.diff(t_s))
    x = np.array([d.pose_x for d in data])
    x = x[:-1]
    y = np.array([d.pose_y for d in data])
    y = y[:-1]
    yaw = np.array([wrapToPi(np.radians(d.pose_yaw)) for d in data])
    yaw = yaw[:-1]
    vx = np.array([d.v_lon for d in data])
    vx = vx[:-1]
    vy = np.array([d.v_lat for d in data])
    vy = vy[:-1]

    # Read control commands
    filename = '../../Data/controls_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_c = np.array([d.time for d in data])
    throttle = np.array([d.throttle for d in data])
    brake = np.array([d.brake for d in data])
    acc = np.array([d.acc for d in data])
    steer = np.array([d.steer * np.pi / 180 for d in data])

    X_OLD = np.vstack((x[:-1], y[:-1], yaw[:-1], vx[:-1], vy[:-1])).T
    X_NEW = np.vstack((x[1:], y[1:], yaw[1:], vx[1:], vy[1:])).T
    U = np.vstack(([acc[:-1], steer[:-1]])).T

    # Unknowns --> [L, p, Cd, Cfr, Ccs]
    params = np.array([4, 10, 0, 0, 0])


    m, n = X_OLD.shape
    J = np.zeros((m * n, len(params)))

    while(1):
        for i in range(m):
            J[i * n:i * n + n, :] = Jacobian(X_OLD[i], U[i], params) * dt

        X_NEW_PRED = predict_new(X_OLD, U, params, dt)
        
        DELTA_X_NEW = X_NEW.flatten() - X_NEW_PRED.flatten()

        delta_params = np.linalg.inv(J.T @ J) @ J.T @ DELTA_X_NEW

        params = params + delta_params

        if np.linalg.norm(delta_params < 0.01):
            break

    print(params)
    np.savetxt('../../Data/params.txt', params)

    start = 0
    pred_states = np.zeros_like(X_NEW[start:, :])

    pred_states[0, :] = np.array([x[start], y[start], yaw[start], vx[start], vy[start]])

    L, p, Cd, Cfr, Ccs = params

    for i in range(1, pred_states.shape[0]):
        x_, y_, yaw_, vx_, vy_ = pred_states[i - 1, :]
        v = np.sqrt(vx_ ** 2 + vy_ ** 2)
        a = U[i + start - 1, 0]
        delta = U[i + start - 1, 1]

        pred_states[i, 0] = x_ + (v * np.cos(np.arctan2(np.tan(delta), 2) + yaw_) * dt)
        pred_states[i, 1] = y_ + (v * np.sin(np.arctan2(np.tan(delta), 2) + yaw_) * dt)
        pred_states[i, 2] = wrapToPi(yaw_ + (v * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
        pred_states[i, 3] = vx_ + (p * a - Cd * v * vx_ - Cfr * vx_) * dt
        pred_states[i, 4] = vy_ - (Ccs * wrapToPi(np.arctan2(vy_, vx_) - delta) + (Cd * v + Cfr) * vy_) * dt

        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.plot(x, y)
    ax1.plot(pred_states[:, 0], pred_states[:, 1])
    ax1.legend(['Actual', 'Predicted'])
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_title('System ID trajectory comparison')

    ax2.plot(range(len(vx)), vx)
    ax2.plot(range(len(pred_states)), pred_states[:, 3])
    ax2.legend(['Actual', 'Predicted'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X velocity')
    ax2.set_title('System ID trajectory comparison')

    ax3.plot(range(len(vy)), vy)
    ax3.plot(range(len(pred_states)), pred_states[:, 4])
    ax3.legend(['Actual', 'Predicted'])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Y velocity')
    ax3.set_title('System ID trajectory comparison')

    ax4.plot(range(len(yaw)), yaw)
    ax4.plot(range(len(pred_states)), pred_states[:, 2])
    ax4.legend(['Actual', 'Predicted'])
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Yaw')
    ax4.set_title('System ID trajectory yaw')
    plt.show()