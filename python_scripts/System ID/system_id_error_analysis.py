#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys
import pickle
import numpy as np
from casadi import *
from numpy.lib.utils import info
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.util import *
from Common.custom_dataclass import *

def predict_new(old_states, control, params, dt):
    L, p, Cd, Cfr, Ccs = params

    x, y, theta, vx, vy = old_states
    v = np.sqrt(vx ** 2 + vy ** 2)
    acc, delta = control
    x_new = x + (v * np.cos(np.arctan2(np.tan(delta), 2) + theta) * dt)
    y_new = y + (v * np.sin(np.arctan2(np.tan(delta), 2) + theta) * dt)
    theta_new = wrapToPi(theta + (v * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
    vx_new = vx + (p * acc - Cd * v * vx - Cfr * vx) * dt
    vy_new = vy - (Ccs * wrapToPi(np.arctan2(vy, vx) - delta) + (Cd * v + Cfr) * vy) * dt
    new_state = np.array([x_new, y_new, theta_new, vx_new, vy_new])

    return new_state


if __name__=='__main__':

    # Read ground truth states
    filename = '../../Data/states_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t = np.array([d.time for d in data])
    t = t[:-1]
    dt = np.average(np.diff(t))

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
    
    states_gt = np.vstack((x, y, yaw, vx, vy)).T
    control = np.vstack((acc, steer)).T

    # start = 00
    # pred_states = np.zeros_like(X_NEW[start:, :])

    # pred_states[0, :] = np.array([x[start], y[start], yaw[start], vx[start], vy[start]])

    # L, p, Cd, Cfr, Ccs = params
    # L, p, q, Cd, Cfr, Ccs = params
    # L, p, h, b, Cd, Cfr, Ccs = params

    # for i in range(1, pred_states.shape[0]):
    #     x_, y_, yaw_, vx_, vy_ = pred_states[i - 1, :]
    #     v = np.sqrt(vx_ ** 2 + vy_ ** 2)
    #     a = U[i + start - 1, 0]
    #     delta = U[i + start - 1, 1]

    #     pred_states[i, 0] = x_ + (np.sqrt(vx_ ** 2 + vy_ ** 2) * np.cos(np.arctan2(np.tan(delta), 2) + yaw_) * dt)
    #     pred_states[i, 1] = y_ + (np.sqrt(vx_ ** 2 + vy_ ** 2) * np.sin(np.arctan2(np.tan(delta), 2) + yaw_) * dt)
    #     pred_states[i, 2] = wrapToPi(yaw_ + (np.sqrt(vx_ ** 2 + vy_ ** 2) * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
    #     pred_states[i, 3] = vx_ + (p * a - Cd * np.sqrt(vx_ ** 2 + vy_ ** 2) * vx_ - Cfr * vx_) * dt
    #     pred_states[i, 4] = vy_ - (Ccs * wrapToPi(np.arctan2(vy_, vx_) - delta) + (Cd * np.sqrt(vx_ ** 2 + vy_ ** 2) + Cfr) * vy_) * dt
        

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # ax1.plot(x, y)
    # ax1.plot(pred_states[:, 0], pred_states[:, 1])
    # ax1.legend(['Actual', 'Predicted'])
    # ax1.set_xlabel('X coordinate')
    # ax1.set_ylabel('Y coordinate')
    # ax1.set_title('System ID trajectory comparison')

    # ax2.plot(range(len(vx)), vx)
    # ax2.plot(range(len(pred_states)), pred_states[:, 3])
    # ax2.legend(['Actual', 'Predicted'])
    # ax2.set_xlabel('Time')
    # ax2.set_ylabel('X velocity')
    # ax2.set_title('System ID trajectory comparison')

    # ax3.plot(range(len(vy)), vy)
    # ax3.plot(range(len(pred_states)), pred_states[:, 4])
    # ax3.legend(['Actual', 'Predicted'])
    # ax3.set_xlabel('Time')
    # ax3.set_ylabel('Y velocity')
    # ax3.set_title('System ID trajectory comparison')

    # ax4.plot(range(len(yaw)), yaw)
    # ax4.plot(range(len(pred_states)), pred_states[:, 2])
    # ax4.legend(['Actual', 'Predicted'])
    # ax4.set_xlabel('Time')
    # ax4.set_ylabel('Yaw')
    # ax4.set_title('System ID trajectory yaw')
    # plt.show()



    horizon = 100
    step = 20

    position_pred = np.zeros((horizon, 2))
    error = np.zeros((int((len(t) - horizon) / step) + 1, horizon))

    params = np.loadtxt('../../Data/params.txt')

    count = 0
    for i in range(0, len(t) - horizon, step):
        curr_state = states_gt[i]
        for j in range(horizon):
            curr_state = predict_new(curr_state, control[i + j], params, t[i + j + 1] - t[i + j])
            error[count, j] = np.sqrt((curr_state[0] - states_gt[i + j + 1, 0])**2 + (curr_state[1] - states_gt[i + j + 1, 1])**2)
        count += 1

    print(count, int((len(t) - horizon) / horizon))
    mean_error = np.median(error, axis=0)
    std_dev_error = np.std(error, axis=0)
    
    plot_1 = plt.figure('Average Position Error over a horizon of {} time steps'.format(horizon))
    plt.title('Average Position Error over a horizon of {} time steps'.format(horizon), fontsize=25)
    plt.xlabel('Time steps', fontsize=15)
    plt.ylabel('Position Error [m]', fontsize=15)
    plt.axis('equal')
    plt.plot([i for i in range(horizon)], mean_error, 'r')
    plt.fill_between([i for i in range(horizon)], mean_error + std_dev_error, mean_error - std_dev_error)
    plt.legend(['Average error', 'Confidence interval - 1 sigma'], fontsize=15)
    plt.show()