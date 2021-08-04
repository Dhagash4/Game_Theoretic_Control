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
    filename = '../Data/GTC/states_1_gtc_diff_speed_ahead.pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_states_1 = np.array([d.time for d in data])
    x_gt_1 = np.array([d.pose_x for d in data])
    y_gt_1 = np.array([d.pose_y for d in data])
    yaw_gt_1 = np.array([d.pose_yaw for d in data])
    v_lon_1 = np.array([d.v_lon for d in data])
    v_lat_1 = np.array([d.v_lat for d in data])

    # Read ground truth states
    filename = '../Data/GTC/states_2_gtc_diff_speed_ahead.pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    t_states_2 = np.array([d.time for d in data])
    x_gt_2 = np.array([d.pose_x for d in data])
    y_gt_2 = np.array([d.pose_y for d in data])
    yaw_gt_2 = np.array([d.pose_yaw for d in data])
    v_lon_2 = np.array([d.v_lon for d in data])
    v_lat_2 = np.array([d.v_lat for d in data])

    # Velocity Plot
    plt.plot(np.arange(len(v_lon_1)), v_lon_1, 'r')
    plt.plot(np.arange(len(v_lon_2)), v_lon_2, 'b')
    plt.legend(['Car 1', 'Car 2'])
    plt.show()


