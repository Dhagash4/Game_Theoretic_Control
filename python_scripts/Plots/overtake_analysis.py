#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# Overtaking Scenarios Analysis

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.plots import *
from Common.util import *


def calculate_nearest_index(current_state: np.ndarray, waypoints: np.ndarray) -> (int):
    """Search nearest waypoint index to the current pose from the waypoints list

    Arguments
    ---------
    - current_state: current state of the vehicle
    - waypoints: pre-computed waypoints to track

    Returns
    -------
    - idx: index of the waypoint corresponding to minimum distance from current pose
    """
    # Extract vehicle position
    x_r, y_r = current_state[:2]
    # Compute squared distance to each waypoint
    dx = (waypoints[:, 0] - x_r) ** 2
    dy = (waypoints[:, 1] - y_r) ** 2
    dist = dx + dy

    # Find index corresponding to minimum distance
    idx = np.argmin(dist)

    return idx

waypoints = read_file("../../Data/2D_waypoints.txt")

car1_states_1_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_1_car1_states.pickle")[:, :5]
car1_states_2_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_2_car1_states.pickle")[:, :5]
car1_states_3_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_3_car1_states.pickle")[:, :5]
car1_states_4_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_4_car1_states.pickle")[:, :5]
car1_states_5_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_5_car1_states.pickle")[:, :5]
car2_states_1_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_1_car2_states.pickle")[:, :5]
car2_states_2_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_2_car2_states.pickle")[:, :5]
car2_states_3_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_3_car2_states.pickle")[:, :5]
car2_states_4_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_4_car2_states.pickle")[:, :5]
car2_states_5_GTC_vs_MPC_1 = read_pickle("../../Data/GTC_vs_MPC/scenario1_5_car2_states.pickle")[:, :5]

car1_states_1_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_1_car1_states.pickle")[:, :5]
car1_states_2_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_2_car1_states.pickle")[:, :5]
car1_states_3_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_3_car1_states.pickle")[:, :5]
car1_states_4_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_4_car1_states.pickle")[:, :5]
car1_states_5_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_5_car1_states.pickle")[:, :5]
car2_states_1_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_1_car2_states.pickle")[:, :5]
car2_states_2_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_2_car2_states.pickle")[:, :5]
car2_states_3_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_3_car2_states.pickle")[:, :5]
car2_states_4_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_4_car2_states.pickle")[:, :5]
car2_states_5_GTC_vs_MPC_11 = read_pickle("../../Data/GTC_vs_MPC/scenario11_5_car2_states.pickle")[:, :5]

car1_states_1_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_1_car1_states.pickle")[:, :5]
car1_states_2_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_2_car1_states.pickle")[:, :5]
car1_states_3_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_3_car1_states.pickle")[:, :5]
car1_states_4_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_4_car1_states.pickle")[:, :5]
car1_states_5_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_5_car1_states.pickle")[:, :5]
car2_states_1_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_1_car2_states.pickle")[:, :5]
car2_states_2_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_2_car2_states.pickle")[:, :5]
car2_states_3_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_3_car2_states.pickle")[:, :5]
car2_states_4_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_4_car2_states.pickle")[:, :5]
car2_states_5_MPC_vs_MPC = read_pickle("../../Data/MPC_vs_MPC/scenario1_5_car2_states.pickle")[:, :5]

car1_states_1_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_1_car1_states.pickle")[:, :5]
car1_states_2_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_2_car1_states.pickle")[:, :5]
car1_states_3_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_3_car1_states.pickle")[:, :5]
car1_states_4_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_4_car1_states.pickle")[:, :5]
car1_states_5_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_5_car1_states.pickle")[:, :5]
car2_states_1_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_1_car2_states.pickle")[:, :5]
car2_states_2_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_2_car2_states.pickle")[:, :5]
car2_states_3_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_3_car2_states.pickle")[:, :5]
car2_states_4_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_4_car2_states.pickle")[:, :5]
car2_states_5_GTC_vs_GTC = read_pickle("../../Data/GTC_vs_GTC/scenario1_5_car2_states.pickle")[:, :5]

idx_1 = np.amin([car1_states_1_GTC_vs_MPC_1.shape[0], car1_states_2_GTC_vs_MPC_1.shape[0], car1_states_3_GTC_vs_MPC_1.shape[0], car1_states_4_GTC_vs_MPC_1.shape[0], car1_states_5_GTC_vs_MPC_1.shape[0]])
idx_2 = np.amin([car1_states_1_MPC_vs_MPC.shape[0], car1_states_2_MPC_vs_MPC.shape[0], car1_states_3_MPC_vs_MPC.shape[0], car1_states_4_MPC_vs_MPC.shape[0], car1_states_5_MPC_vs_MPC.shape[0]])
idx_3 = np.amin([car1_states_1_GTC_vs_GTC.shape[0], car1_states_2_GTC_vs_GTC.shape[0], car1_states_3_GTC_vs_GTC.shape[0], car1_states_4_GTC_vs_GTC.shape[0], car1_states_5_GTC_vs_GTC.shape[0]])
idx_4 = np.amin([car1_states_1_GTC_vs_MPC_11.shape[0], car1_states_2_GTC_vs_MPC_11.shape[0], car1_states_3_GTC_vs_MPC_11.shape[0], car1_states_4_GTC_vs_MPC_11.shape[0], car1_states_5_GTC_vs_MPC_11.shape[0]])


car1_states_all_GTC_vs_MPC_1 = np.dstack((car1_states_1_GTC_vs_MPC_1[:idx_1], car1_states_2_GTC_vs_MPC_1[:idx_1], car1_states_3_GTC_vs_MPC_1[:idx_1], car1_states_4_GTC_vs_MPC_1[:idx_1], car1_states_5_GTC_vs_MPC_1[:idx_1]))
car2_states_all_GTC_vs_MPC_1 = np.dstack((car2_states_1_GTC_vs_MPC_1[:idx_1], car2_states_2_GTC_vs_MPC_1[:idx_1], car2_states_3_GTC_vs_MPC_1[:idx_1], car2_states_4_GTC_vs_MPC_1[:idx_1], car2_states_5_GTC_vs_MPC_1[:idx_1]))

car1_states_all_GTC_vs_MPC_11 = np.dstack((car1_states_1_GTC_vs_MPC_11[:idx_4], car1_states_2_GTC_vs_MPC_11[:idx_4], car1_states_3_GTC_vs_MPC_11[:idx_4], car1_states_4_GTC_vs_MPC_11[:idx_4], car1_states_5_GTC_vs_MPC_11[:idx_4]))
car2_states_all_GTC_vs_MPC_11 = np.dstack((car2_states_1_GTC_vs_MPC_11[:idx_4], car2_states_2_GTC_vs_MPC_11[:idx_4], car2_states_3_GTC_vs_MPC_11[:idx_4], car2_states_4_GTC_vs_MPC_11[:idx_4], car2_states_5_GTC_vs_MPC_11[:idx_4]))

car1_states_all_MPC_vs_MPC = np.dstack((car1_states_1_MPC_vs_MPC[:idx_2], car1_states_2_MPC_vs_MPC[:idx_2], car1_states_3_MPC_vs_MPC[:idx_2], car1_states_4_MPC_vs_MPC[:idx_2], car1_states_5_MPC_vs_MPC[:idx_2]))
car2_states_all_MPC_vs_MPC = np.dstack((car2_states_1_MPC_vs_MPC[:idx_2], car2_states_2_MPC_vs_MPC[:idx_2], car2_states_3_MPC_vs_MPC[:idx_2], car2_states_4_MPC_vs_MPC[:idx_2], car2_states_5_MPC_vs_MPC[:idx_2]))

car1_states_all_GTC_vs_GTC = np.dstack((car1_states_1_GTC_vs_GTC[:idx_3], car1_states_2_GTC_vs_GTC[:idx_3], car1_states_3_GTC_vs_GTC[:idx_3], car1_states_4_GTC_vs_GTC[:idx_3], car1_states_5_GTC_vs_GTC[:idx_3]))
car2_states_all_GTC_vs_GTC = np.dstack((car2_states_1_GTC_vs_GTC[:idx_3], car2_states_2_GTC_vs_GTC[:idx_3], car2_states_3_GTC_vs_GTC[:idx_3], car2_states_4_GTC_vs_GTC[:idx_3], car2_states_5_GTC_vs_GTC[:idx_3]))

nearest_idx_GTC_vs_MPC_1 = np.zeros((idx_1, 5, 2))
nearest_idx_MPC_vs_MPC = np.zeros((idx_2, 5, 2))
nearest_idx_GTC_vs_GTC = np.zeros((idx_3, 5, 2))
nearest_idx_GTC_vs_MPC_11 = np.zeros((idx_4, 5, 2))

for d in range(5):
    for i in range(idx_1):
        nearest_idx_GTC_vs_MPC_1[i, d, 0] = calculate_nearest_index(car1_states_all_GTC_vs_MPC_1[i, :, d], waypoints)
        nearest_idx_GTC_vs_MPC_1[i, d, 1] = calculate_nearest_index(car2_states_all_GTC_vs_MPC_1[i, :, d], waypoints)

    for i in range(idx_2):
        nearest_idx_MPC_vs_MPC[i, d, 0] = calculate_nearest_index(car1_states_all_MPC_vs_MPC[i, :, d], waypoints)
        nearest_idx_MPC_vs_MPC[i, d, 1] = calculate_nearest_index(car2_states_all_MPC_vs_MPC[i, :, d], waypoints)

    for i in range(idx_3):
        nearest_idx_GTC_vs_GTC[i, d, 0] = calculate_nearest_index(car1_states_all_GTC_vs_GTC[i, :, d], waypoints)
        nearest_idx_GTC_vs_GTC[i, d, 1] = calculate_nearest_index(car2_states_all_GTC_vs_GTC[i, :, d], waypoints)
    
    for i in range(idx_4):
        nearest_idx_GTC_vs_MPC_11[i, d, 0] = calculate_nearest_index(car1_states_all_GTC_vs_MPC_11[i, :, d], waypoints)
        nearest_idx_GTC_vs_MPC_11[i, d, 1] = calculate_nearest_index(car2_states_all_GTC_vs_MPC_11[i, :, d], waypoints)

overtake_idx_GTC_vs_MPC_1 = np.zeros(5)
overtake_idx_GTC_vs_MPC_11 = np.zeros(5)
overtake_idx_MPC_vs_MPC = np.zeros(5)
overtake_idx_GTC_vs_GTC = np.zeros(5)

for d in range(5):
    try:
        overtake_idx_GTC_vs_MPC_1[d] = np.amin(np.where(nearest_idx_GTC_vs_MPC_1[:, d, 0] < nearest_idx_GTC_vs_MPC_1[:, d, 1]))
    except ValueError:
        overtake_idx_GTC_vs_MPC_1[d] = idx_1
    
    try:
        overtake_idx_GTC_vs_MPC_11[d] = np.amin(np.where(nearest_idx_GTC_vs_MPC_11[:, d, 0] > nearest_idx_GTC_vs_MPC_11[:, d, 1]))
    except ValueError:
        overtake_idx_GTC_vs_MPC_11[d] = idx_4

    try:
        overtake_idx_MPC_vs_MPC[d] = np.amin(np.where(nearest_idx_MPC_vs_MPC[:, d, 0] < nearest_idx_MPC_vs_MPC[:, d, 1]))
    except ValueError:
        overtake_idx_MPC_vs_MPC[d] = idx_2
    
    try:
        overtake_idx_GTC_vs_GTC[d] = np.amin(np.where(nearest_idx_GTC_vs_GTC[:, d, 0] < nearest_idx_GTC_vs_GTC[:, d, 1]))
    except ValueError:
        overtake_idx_GTC_vs_GTC[d] = idx_3

print(overtake_idx_GTC_vs_MPC_1, overtake_idx_GTC_vs_MPC_11, overtake_idx_MPC_vs_MPC, overtake_idx_GTC_vs_GTC)

f, ax = plt.subplots()
plt.suptitle('Analysis of overtaking time', fontsize=20)
ax.scatter(np.ones_like(overtake_idx_GTC_vs_MPC_1) * 0.1, overtake_idx_GTC_vs_MPC_1 * 0.1, alpha=0.5)
ax.scatter(np.ones_like(overtake_idx_GTC_vs_MPC_11) * 0.2, overtake_idx_GTC_vs_MPC_11 * 0.1, alpha=0.5)
ax.scatter(np.ones_like(overtake_idx_MPC_vs_MPC) * 0.3, overtake_idx_MPC_vs_MPC * 0.1, alpha=0.5)
ax.scatter(np.ones_like(overtake_idx_GTC_vs_GTC) * 0.4, overtake_idx_GTC_vs_GTC * 0.1, alpha=0.5)
ax.set_xticks([0.1, 0.2, 0.3, 0.4])
ax.set_xticklabels(['GTC v/s MPC\nGTC car overatakes MPC car', 'GTC v/s MPC\nMPC car overtakes GTC car', 'MPC v/s MPC', 'GTC v/s GTC',], fontsize=15)
ax.set_ylabel('Time to overtake (simulation seconds)', fontsize=15)
plt.text(0.5, 1.05, '(The overtaking car starts behind the opponent with higher maximum velocity)', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=20, color='red')

plt.show()