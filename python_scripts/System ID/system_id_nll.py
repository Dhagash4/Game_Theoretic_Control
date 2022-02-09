#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# System Identification using Non-Linear Least Squares Approach

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


def Jacobian(states: np.ndarray, control: np.ndarray, sys_params: np.ndarray) -> (np.ndarray):
    """Compute the Jacobian matrix for the chosen state transition equations (dynamic model of the vehicles)

    Arguments
    ---------
    - states: states of the vehicle at which Jacobian has to be evaluated [x, y, yaw, vx, vy]
    - control: control commands applied to the vehicle at that instant
    - sys_params: estimate of the unknown model parameters

    Returns
    -------
    - J: Jacobian matrix evaluated at given state and controls with the rough estimate of model parameters
    """
    _, _, _, vx, vy = states
    acc, delta = control
    L, _, _, _, _ = sys_params    
    
    dtheta_dL = (-np.sqrt(vx ** 2 + vy ** 2) * np.tan(delta) * np.cos(np.arctan2(np.tan(delta), 2))) / (L ** 2)

    dvx_dp = acc
    dvx_dCd = -1 * np.sqrt(vx ** 2 + vy ** 2) * vx
    dvx_dCfr = -1 * vx

    dvy_dCd = -1 * np.sqrt(vx ** 2 + vy ** 2) * vy
    dvy_dCfr = -1 * vy
    dvy_dCcs = -wrapToPi(np.arctan2(vy, vx) - delta)

    J = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [dtheta_dL, 0, 0, 0, 0],
                  [0, dvx_dp, dvx_dCd, dvx_dCfr, 0],
                  [0, 0, dvy_dCd, dvy_dCfr, dvy_dCcs]])

    return J

def predict_new(old_states, control, sys_params, dt):
    """Generates a prediction of the new vehicle states given the old state and the controls applied
    
    Arguments
    ---------
    - old_states: old state of the vehicle [x, y, yaw, vx, vy]
    - control: control commands given to the vehicle [throttle/brake, steer]
    - sys_params: vehicle dynamics parameters obtained from System ID
    - dt: sampling time of the controller

    Returns
    -------
    - new_state: new predicted state of the vehicle [x, y, yaw, vx, vy]
    """
    L, p, Cd, Cfr, Ccs = sys_params
    new_states = np.zeros_like(old_states)

    m, n = old_states.shape
    for i in range(m):
        x, y, theta, vx, vy = old_states[i]
        acc, delta = control[i]
        x_new = x + (np.sqrt(vx ** 2 + vy ** 2) * np.cos(np.arctan2(np.tan(delta), 2) + theta) * dt)
        y_new = y + (np.sqrt(vx ** 2 + vy ** 2) * np.sin(np.arctan2(np.tan(delta), 2) + theta) * dt)
        theta_new = wrapToPi(theta + (np.sqrt(vx ** 2 + vy ** 2) * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
        vx_new = vx + (p * acc - Cd * np.sqrt(vx ** 2 + vy ** 2) * vx - Cfr * vx) * dt
        vy_new = vy - (Ccs * wrapToPi(np.arctan2(vy, vx) - delta) + (Cd * np.sqrt(vx ** 2 + vy ** 2) + Cfr) * vy) * dt
        new_states[i] = [x_new, y_new, theta_new, vx_new, vy_new]

    return new_states

if __name__=='__main__':
    # Read ground truth states
    filename = '../../Data/states_sys_id.pickle'
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
    filename = '../../Data/controls_sys_id.pickle'
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

    start = 200
    end = 500
    pred_states = np.zeros_like(X_NEW[start:, :])

    pred_states[start, :] = np.array([x[start], y[start], yaw[start], vx[start], vy[start]])

    L, p, Cd, Cfr, Ccs = params

    for i in range(start + 1, end + 1):
        x_, y_, yaw_, vx_, vy_ = pred_states[i - 1, :]
        v = np.sqrt(vx_ ** 2 + vy_ ** 2)
        a = U[i - 1, 0]
        delta = U[i - 1, 1]
        dt = t_c[i] - t_c[i - 1]
        pred_states[i, 0] = x_ + (np.sqrt(vx_ ** 2 + vy_ ** 2) * np.cos(np.arctan2(np.tan(delta), 2) + yaw_) * dt)
        pred_states[i, 1] = y_ + (np.sqrt(vx_ ** 2 + vy_ ** 2) * np.sin(np.arctan2(np.tan(delta), 2) + yaw_) * dt)
        pred_states[i, 2] = wrapToPi(yaw_ + (np.sqrt(vx_ ** 2 + vy_ ** 2) * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
        pred_states[i, 3] = vx_ + (p * a - Cd * np.sqrt(vx_ ** 2 + vy_ ** 2) * vx_ - Cfr * vx_) * dt
        pred_states[i, 4] = vy_ - (Ccs * wrapToPi(np.arctan2(vy_, vx_) - delta) + (Cd * np.sqrt(vx_ ** 2 + vy_ ** 2) + Cfr) * vy_) * dt       

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(x[start + 1:end], y[start + 1:end])
    ax1.plot(pred_states[start + 1:end, 0], pred_states[start + 1:end, 1])
    ax1.legend(['Actual', 'Predicted'])
    ax1.set_xlabel('X coordinate [m]')
    ax1.set_ylabel('Y coordinate [m]')
    ax1.set_title('System ID trajectory prediction')

    ax2.plot(range(len(yaw[start + 1:end])), yaw[start + 1:end])
    ax2.plot(range(len(pred_states[start + 1:end])), pred_states[start + 1:end, 2])
    ax2.legend(['Actual', 'Predicted'])
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Yaw [radians]')
    ax2.set_title('System ID trajectory yaw')

    plt.show()