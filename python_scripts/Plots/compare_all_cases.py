#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# Comparison of trajectories

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys, os
import numpy as np
import shutil
import imageio
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.plots import *
from Common.util import *

waypoints = read_file("../../Data/2D_waypoints.txt")

car1_states_1_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_1_car1_states.pickle")[:, :5]
car1_states_2_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_2_car1_states.pickle")[:, :5]
car1_states_3_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_3_car1_states.pickle")[:, :5]
car1_states_4_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_4_car1_states.pickle")[:, :5]
car1_states_5_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_5_car1_states.pickle")[:, :5]
car2_states_1_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_1_car2_states.pickle")[:, :5]
car2_states_2_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_2_car2_states.pickle")[:, :5]
car2_states_3_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_3_car2_states.pickle")[:, :5]
car2_states_4_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_4_car2_states.pickle")[:, :5]
car2_states_5_GTC_vs_MPC = read_pickle("../../Data/GTC_vs_MPC/scenario1_5_car2_states.pickle")[:, :5]

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

idx_1 = np.amin([car1_states_1_GTC_vs_MPC.shape[0], car1_states_2_GTC_vs_MPC.shape[0], car1_states_3_GTC_vs_MPC.shape[0], car1_states_4_GTC_vs_MPC.shape[0], car1_states_5_GTC_vs_MPC.shape[0]])
idx_2 = np.amin([car1_states_1_MPC_vs_MPC.shape[0], car1_states_2_MPC_vs_MPC.shape[0], car1_states_3_MPC_vs_MPC.shape[0], car1_states_4_MPC_vs_MPC.shape[0], car1_states_5_MPC_vs_MPC.shape[0]])
idx_3 = np.amin([car1_states_1_GTC_vs_GTC.shape[0], car1_states_2_GTC_vs_GTC.shape[0], car1_states_3_GTC_vs_GTC.shape[0], car1_states_4_GTC_vs_GTC.shape[0], car1_states_5_GTC_vs_GTC.shape[0]])

car1_states_all_GTC_vs_MPC = np.dstack((car1_states_1_GTC_vs_MPC[:idx_1], car1_states_2_GTC_vs_MPC[:idx_1], car1_states_3_GTC_vs_MPC[:idx_1], car1_states_4_GTC_vs_MPC[:idx_1], car1_states_5_GTC_vs_MPC[:idx_1]))
car2_states_all_GTC_vs_MPC = np.dstack((car2_states_1_GTC_vs_MPC[:idx_1], car2_states_2_GTC_vs_MPC[:idx_1], car2_states_3_GTC_vs_MPC[:idx_1], car2_states_4_GTC_vs_MPC[:idx_1], car2_states_5_GTC_vs_MPC[:idx_1]))

car1_states_all_MPC_vs_MPC = np.dstack((car1_states_1_MPC_vs_MPC[:idx_2], car1_states_2_MPC_vs_MPC[:idx_2], car1_states_3_MPC_vs_MPC[:idx_2], car1_states_4_MPC_vs_MPC[:idx_2], car1_states_5_MPC_vs_MPC[:idx_2]))
car2_states_all_MPC_vs_MPC = np.dstack((car2_states_1_MPC_vs_MPC[:idx_2], car2_states_2_MPC_vs_MPC[:idx_2], car2_states_3_MPC_vs_MPC[:idx_2], car2_states_4_MPC_vs_MPC[:idx_2], car2_states_5_MPC_vs_MPC[:idx_2]))

car1_states_all_GTC_vs_GTC = np.dstack((car1_states_1_GTC_vs_GTC[:idx_3], car1_states_2_GTC_vs_GTC[:idx_3], car1_states_3_GTC_vs_GTC[:idx_3], car1_states_4_GTC_vs_GTC[:idx_3], car1_states_5_GTC_vs_GTC[:idx_3]))
car2_states_all_GTC_vs_GTC = np.dstack((car2_states_1_GTC_vs_GTC[:idx_3], car2_states_2_GTC_vs_GTC[:idx_3], car2_states_3_GTC_vs_GTC[:idx_3], car2_states_4_GTC_vs_GTC[:idx_3], car2_states_5_GTC_vs_GTC[:idx_3]))

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
f.suptitle('Comparison Scenario: Car 1 starts ahead with lower maximum speed', fontsize=15)

mng = plt.get_current_fig_manager()
mng.window.showMaximized()

plt.tight_layout()

try:
    os.mkdir('temp/')
except FileExistsError:
    shutil.rmtree('temp/')
    os.mkdir('temp/')

color = ['black', 'blue', 'green', 'orange', 'gray']
            
l = np.amin(np.r_[idx_1, idx_2, idx_3])

for i in range(l):
    ax1.set_title('GTC v/s MPC', fontsize=12)
    ax2.set_title('MPC v/s MPC', fontsize=12)
    ax3.set_title('GTC v/s GTC', fontsize=12)

    plot_track_boundaries(ax1, waypoints[1150:1350], 15)
    plot_track_boundaries(ax2, waypoints[1150:1350], 15)
    plot_track_boundaries(ax3, waypoints[1150:1350], 15)

    ax1.scatter(car1_states_all_GTC_vs_MPC[i, 0, 0], car1_states_all_GTC_vs_MPC[i, 1, 0], marker='o', label='Car 1 (GTC)', color=color[0])
    ax1.plot(car1_states_all_GTC_vs_MPC[:i+1, 0, 0], car1_states_all_GTC_vs_MPC[:i+1, 1, 0], color[0], label='Test Run {}'.format(1))
    ax2.scatter(car1_states_all_MPC_vs_MPC[i, 0, 0], car1_states_all_MPC_vs_MPC[i, 1, 0], marker='o', label='Car 1', color=color[0])
    ax2.plot(car1_states_all_MPC_vs_MPC[:i+1, 0, 0], car1_states_all_MPC_vs_MPC[:i+1, 1, 0], color[0], label='Test Run {}'.format(1))
    ax3.scatter(car1_states_all_GTC_vs_GTC[i, 0, 0], car1_states_all_GTC_vs_GTC[i, 1, 0], marker='o', label='Car 1', color=color[0])
    ax3.plot(car1_states_all_GTC_vs_GTC[:i+1, 0, 0], car1_states_all_GTC_vs_GTC[:i+1, 1, 0], color[0], label='Test Run {}'.format(1))
    for j in range(1, 5):
        ax1.scatter(car1_states_all_GTC_vs_MPC[i, 0, j], car1_states_all_GTC_vs_MPC[i, 1, j], marker='o', color=color[j])
        ax1.plot(car1_states_all_GTC_vs_MPC[:i+1, 0, j], car1_states_all_GTC_vs_MPC[:i+1, 1, j], color[j], label='Test Run {}'.format(j+1))
        ax2.scatter(car1_states_all_MPC_vs_MPC[i, 0, j], car1_states_all_MPC_vs_MPC[i, 1, j], marker='o', color=color[j])
        ax2.plot(car1_states_all_MPC_vs_MPC[:i+1, 0, j], car1_states_all_MPC_vs_MPC[:i+1, 1, j], color[j], label='Test Run {}'.format(j+1))
        ax3.scatter(car1_states_all_GTC_vs_GTC[i, 0, j], car1_states_all_GTC_vs_GTC[i, 1, j], marker='o', color=color[j])
        ax3.plot(car1_states_all_GTC_vs_GTC[:i+1, 0, j], car1_states_all_GTC_vs_GTC[:i+1, 1, j], color[j], label='Test Run {}'.format(j+1))

    ax1.scatter(car2_states_all_GTC_vs_MPC[i, 0, 0], car2_states_all_GTC_vs_MPC[i, 1, 0], marker='x', label='Car 2 (MPC)', color=color[0])
    ax1.plot(car2_states_all_GTC_vs_MPC[:i+1, 0, 0], car2_states_all_GTC_vs_MPC[:i+1, 1, 0], color[0])
    ax2.scatter(car2_states_all_MPC_vs_MPC[i, 0, 0], car2_states_all_MPC_vs_MPC[i, 1, 0], marker='x', label='Car 2', color=color[0])
    ax2.plot(car2_states_all_MPC_vs_MPC[:i+1, 0, 0], car2_states_all_MPC_vs_MPC[:i+1, 1, 0], color[0])
    ax3.scatter(car2_states_all_GTC_vs_GTC[i, 0, 0], car2_states_all_GTC_vs_GTC[i, 1, 0], marker='x', label='Car 2', color=color[0])
    ax3.plot(car2_states_all_GTC_vs_GTC[:i+1, 0, 0], car2_states_all_GTC_vs_GTC[:i+1, 1, 0], color[0])
    for j in range(1, 5):
        ax1.scatter(car2_states_all_GTC_vs_MPC[i, 0, j], car2_states_all_GTC_vs_MPC[i, 1, j], marker='x', color=color[j])
        ax1.plot(car2_states_all_GTC_vs_MPC[:i+1, 0, j], car2_states_all_GTC_vs_MPC[:i+1, 1, j], color[j])
        ax2.scatter(car2_states_all_MPC_vs_MPC[i, 0, j], car2_states_all_MPC_vs_MPC[i, 1, j], marker='x', color=color[j])
        ax2.plot(car2_states_all_MPC_vs_MPC[:i+1, 0, j], car2_states_all_MPC_vs_MPC[:i+1, 1, j], color[j])
        ax3.scatter(car2_states_all_GTC_vs_GTC[i, 0, j], car2_states_all_GTC_vs_GTC[i, 1, j], marker='x', color=color[j])
        ax3.plot(car2_states_all_GTC_vs_GTC[:i+1, 0, j], car2_states_all_GTC_vs_GTC[:i+1, 1, j], color[j])

    ax1.legend(fontsize=12)
    ax2.legend(fontsize=12)
    ax3.legend(fontsize=12)

    plt.pause(0.1)
    plt.savefig('temp/{}.png'.format(i+1))
    ax1.clear()
    ax2.clear()
    ax3.clear()

with imageio.get_writer('../../GIFS/Compare/scene1.gif', mode='I') as writer:
    for filename in ['temp/{}.png'.format(i+1) for i in range(l)]:
        image = imageio.imread(filename)
        writer.append_data(image)
shutil.rmtree('temp/')

plt.close()