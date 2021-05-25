#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import pickle
import numpy as np
from matplotlib import pyplot as plt

from Common.custom_dataclass import *

if __name__=='__main__':

    # Desired Trajectory
    filename = '../Data/2D_waypoints.txt'
    data = np.loadtxt(filename)
    x_des = data[:, 0]
    y_des = data[:, 1]

    # Read ground truth states
    filename = '../Data/states_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_states = np.array([d.time for d in data])
    x_gt = np.array([d.pose_x for d in data])
    y_gt = np.array([d.pose_y for d in data])
    yaw_gt = np.array([d.pose_yaw for d in data])
    v_lon = np.array([d.v_lon for d in data])
    v_lat = np.array([d.v_lat for d in data])

    # Read control commands
    filename = '../Data/controls_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_controls = np.array([d.time for d in data])
    throttle = np.array([d.throttle for d in data])
    brake = np.array([d.brake for d in data])
    steer = np.array([d.steer for d in data])

    # Read tracking errors
    filename = '../Data/errors_e(0.100000)_v(5.000000)_p(0.500000)_i(0.010000)_d(0.150000)_n(1.000000).pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_errors = np.array([d.time for d in data])
    heading_error = np.array([d.heading_error for d in data])
    crosstrack_error = np.array([d.crosstrack_error for d in data])

    # Velocity Plot
    plt.figure('Velocity Control')
    plt.plot(t_states - t_states[0], v_lon, 'g')
    plt.plot(t_states - t_states[0], v_lat, 'r')
    plt.axhline(14)
    plt.xlabel("Time [$s$]", fontsize='large')
    plt.ylabel("Velocity [$m/s$]", fontsize='large')
    plt.title("Velocity Tracking", fontsize='large')
    plt.legend(['Ground truth longitudinal velocity', 'Ground truth lateral velocity', 'Setpoint longitudinal velocity'], fontsize='large')
    plt.show()

    # Heading Error Plot
    plt.figure('Crosstrack Error')
    plt.plot(t_errors - t_errors[0], heading_error, 'r')
    plt.xlabel("Time [$s$]", fontsize='large')
    plt.ylabel("Heading error [$radians$]", fontsize='large')
    plt.title("Trajectory Tracking [Heading]", fontsize='large')
    plt.show()

    # Trajectory Plot
    plt.figure('Trajectory')
    plt.plot(x_gt, y_gt, 'b-')
    plt.plot(x_des, y_des, 'g-')
    plt.plot(x_des[0], y_des[0], 'ro')
    plt.xlabel('x coordinate [$m$]')
    plt.ylabel('y coordinate [$m$]')
    plt.legend(['Ground Truth', 'Desired', 'Start Point'])
    plt.title('Vehicle Trajectory')
    plt.show()

