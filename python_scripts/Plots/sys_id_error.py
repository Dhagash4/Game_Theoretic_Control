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

    mean_error = np.median(error, axis=0)
    std_dev_error = np.std(error, axis=0)
    
    h = [i for i in range(horizon)]

    f = plt.figure('Average Position Error over a horizon of {} time steps'.format(horizon))
    plt.tight_layout()
    plt.title('Average Position Error over a horizon of {} time steps'.format(horizon), fontsize=25)
    ax = plt.subplot(2, 2, 1)
    ax.set_xlabel('Time steps', fontsize=15)
    ax.set_ylabel('Position Error [m]', fontsize=15)
    for i in range(error.shape[0]):
        ax.scatter(h[:25], error[i, :25], s=1, c='b')
    violin_parts = ax.violinplot(error[:, :25], positions=h[:25], showmedians=True)
    for partname in ('cmins','cmaxes'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    vp = violin_parts['cmedians']
    vp.set_edgecolor('yellow')
    vp.set_linewidth(1)
    for vp in violin_parts['bodies']:
        vp.set_facecolor('red')
        vp.set_edgecolor('black')
        vp.set_linewidth(2)
        vp.set_alpha(0.5)

    ax = plt.subplot(2, 2, 2)
    ax.set_xlabel('Time steps', fontsize=15)
    ax.set_ylabel('Position Error [m]', fontsize=15)
    for i in range(error.shape[0]):
        ax.scatter(h[25:50], error[i, 25:50], s=1, c='b')
    violin_parts = ax.violinplot(error[:, 25:50], positions=h[25:50], showmedians=True)
    for partname in ('cmins','cmaxes'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    vp = violin_parts['cmedians']
    vp.set_edgecolor('yellow')
    vp.set_linewidth(1)
    for vp in violin_parts['bodies']:
        vp.set_facecolor('red')
        vp.set_edgecolor('black')
        vp.set_linewidth(2)
        vp.set_alpha(0.5)

    ax = plt.subplot(2, 2, 3)
    ax.set_xlabel('Time steps', fontsize=15)
    ax.set_ylabel('Position Error [m]', fontsize=15)
    for i in range(error.shape[0]):
        ax.scatter(h[50:75], error[i, 50:75], s=1, c='b')
    violin_parts = ax.violinplot(error[:, 50:75], positions=h[50:75], showmedians=True)
    for partname in ('cmins','cmaxes'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    vp = violin_parts['cmedians']
    vp.set_edgecolor('yellow')
    vp.set_linewidth(2)
    for vp in violin_parts['bodies']:
        vp.set_facecolor('red')
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
        vp.set_alpha(0.5)

    ax = plt.subplot(2, 2, 4)
    ax.set_xlabel('Time steps', fontsize=15)
    ax.set_ylabel('Position Error [m]', fontsize=15)
    for i in range(error.shape[0]):
        ax.scatter(h[75:], error[i, 75:], s=1, c='b')
    violin_parts = ax.violinplot(error[:, 75:], positions=h[75:], showmedians=True)
    for partname in ('cmins','cmaxes'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    vp = violin_parts['cmedians']
    vp.set_edgecolor('yellow')
    vp.set_linewidth(2)
    for vp in violin_parts['bodies']:
        vp.set_facecolor('red')
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    plt.show()