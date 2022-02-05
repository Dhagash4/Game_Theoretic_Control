#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# System Identification Error Analysis

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys
import pickle
import numpy as np
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

    acc = np.array([d.acc for d in data])
    steer = np.array([d.steer * np.pi / 180 for d in data])
    
    states_gt = np.vstack((x, y, yaw, vx, vy)).T
    control = np.vstack((acc, steer)).T

    horizon = 100
    step = 10

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

    median_error = np.median(error, axis=0)
    q10, q25, q40, q60, q75, q90 = np.percentile(error, [10, 25, 40, 60, 75, 90], axis=0)
    
    h = np.array([i for i in range(horizon)]) * 0.1

    f = plt.figure('Average Position Error over a horizon of {} time steps'.format(horizon))
    plt.tight_layout()
    plt.yscale('log')
    plt.title('System ID error analysis over prediction horizon of 10 seconds', fontsize=20)
    ax = plt.subplot(1, 1, 1)
    ax.scatter(h, error[0], s=1, c='blue', alpha=0.1, label='Raw Error Values')
    for i in range(1, error.shape[0]):
        ax.scatter(h, error[i], s=1, c='darkblue', alpha=0.2)
    ax.plot(h, median_error, color='black', label='Median of Error Values')
    ax.fill_between(h, q10, q90, color='lightgray', alpha=0.3, label='10-90% Interquartile Range')
    ax.fill_between(h, q25, q75, color='darkgray', alpha=0.3, label='25-75% Interquartile Range')
    ax.fill_between(h, q40, q60, color='dimgray', alpha=0.3, label='40-60% Interquartile Range')
    ax.set_xlabel('Prediction horizon in seconds', fontsize=15)
    ax.set_ylabel('Error between predicted and ground truth position [m] - Log Scale', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()