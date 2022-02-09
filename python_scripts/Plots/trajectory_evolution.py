#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# Create GIF for trajectory evolution

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

def plot_trajectory_evolution(ax: plt.axes, states_car_1: np.ndarray, states_car_2: np.ndarray, title: str, gifname: str) -> (plt.axes):
    try:
        os.mkdir('temp/')
    except FileExistsError:
        shutil.rmtree('temp/')
        os.mkdir('temp/')

    color = [['red', 'deeppink', 'orange', 'chocolate', 'brown'],
             ['cyan', 'lime', 'navy', 'darkgreen', 'blue']]
            
    l = states_car_1.shape[0]
    if len(states_car_1.shape) == 3:
        d = states_car_1.shape[2]
        for i in range(10, l):
            plt.title(title, fontsize=18)
            plt.text(0.01, 0.65, 'Car 2 starts ahead with lower maximum speed', bbox=dict(facecolor='red', alpha=0.5),
             transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontsize=12)
            plot_track_boundaries(ax, waypoints[105:280], 15)
            # plot_track_boundaries(ax, waypoints[1150:1400], 15)
            # plot_track_boundaries(ax, waypoints[1150:1350], 15)
            for j in range(d):
                ax.plot(states_car_1[i, 0, j], states_car_1[i, 1, j], marker='o', color=color[0][j], mfc='none')
                ax.plot(states_car_1[:i+1, 0, j], states_car_1[:i+1, 1, j], color[0][j], label='GTC Car: Test Run {}'.format(j+1))
            for j in range(d):
                ax.plot(states_car_2[i, 0, j], states_car_2[i, 1, j], marker='o', color=color[1][j], mfc='none')
                ax.plot(states_car_2[:i+1, 0, j], states_car_2[:i+1, 1, j], color[1][j], label='MPC Car: Test Run {}'.format(j+1))
            plt.legend(fontsize=12)
            plt.pause(0.1)
            plt.savefig('temp/{}.png'.format(i+1))
            ax.clear()
    else:
        for i in range(l):
            traj += ax.plot(states_car_1[i, 0], states_car_1[i, 1], 'rx')
            traj += ax.plot(states_car_2[i, 0], states_car_2[i, 1], 'rx')
            traj = ax.plot(states_car_1[:i+1, 0], states_car_1[:i+1, 1], 'g')
            traj += ax.plot(states_car_2[:i+1, 0], states_car_2[:i+1, 1], 'b')
            plt.legend(traj[:2], fontsize=15)
            plt.pause(0.01)
            plt.savefig('temp/{}.png'.format(i+1))
            traj.clear()

    # Generate GIF
    with imageio.get_writer(gifname, mode='I') as writer:
        for filename in ['temp/{}.png'.format(i+1) for i in range(l)]:
            image = imageio.imread(filename)
            writer.append_data(image)
    shutil.rmtree('temp/')
    return ax

def plot_velocity_evolution(ax: plt.axes, states_car_1: np.ndarray, states_car_2: np.ndarray, title: str, legend: list, gifname: str) -> (plt.axes):
    try:
        os.mkdir('temp/')
    except FileExistsError:
        shutil.rmtree('temp/')
        os.mkdir('temp/')

    l = states_car_1.shape[0]
    ax.set_xlabel('Time steps', fontsize=15)
    ax.set_ylabel('Velocity [m/s]', fontsize=15)
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 25])
    plt.title(title, fontsize=15)
    plt.tight_layout()
    for i in range(l):
        vel = ax.plot(np.arange(i+1), states_car_1[:i+1, 3], 'g')
        vel += ax.plot(np.arange(i+1), states_car_2[:i+1, 3], 'b')
        plt.legend(vel, legend, fontsize=15)
        plt.pause(0.01)
        plt.savefig('temp/{}.png'.format(i+1))
        vel.clear()

    # Generate GIF
    with imageio.get_writer(gifname, mode='I') as writer:
        for filename in ['temp/{}.png'.format(i+1) for i in range(l)]:
            image = imageio.imread(filename)
            writer.append_data(image)
    shutil.rmtree('temp/')
    return ax

def plot_trajectory_velocity_evolution(ax1: plt.axes, ax2: plt.axes, states_car_1: np.ndarray, states_car_2: np.ndarray, titles: list, legend: list, gifname: str) -> (plt.axes):
    try:
        os.mkdir('temp/')
    except FileExistsError:
        shutil.rmtree('temp/')
        os.mkdir('temp/')
    
    l = states_car_1.shape[0]
    ax1.set_title(titles[0], fontsize=10)
    ax2.set_title(titles[1], fontsize=10)
    ax2.set_xlabel('Time steps', fontsize=10)
    ax2.set_ylabel('Velocity [m/s]', fontsize=10)
    ax2.set_xlim([0, 250])
    ax2.set_ylim([0, 25])
    plt.tight_layout()
    for i in range(l):
        traj = ax1.plot(states_car_1[:i+1, 0], states_car_1[:i+1, 1], 'g')
        traj += ax1.plot(states_car_2[:i+1, 0], states_car_2[:i+1, 1], 'b')
        plt.legend(traj, legend, fontsize=10)

        vel = ax2.plot(np.arange(i+1), states_car_1[:i+1, 3], 'g')
        vel += ax2.plot(np.arange(i+1), states_car_2[:i+1, 3], 'b')
        plt.legend(vel, legend, fontsize=10)
        plt.pause(0.01)
        plt.savefig('temp/{}.png'.format(i+1))
        traj.clear()
        vel.clear()

    # Generate GIF
    with imageio.get_writer(gifname, mode='I') as writer:
        for filename in ['temp/{}.png'.format(i+1) for i in range(l)]:
            image = imageio.imread(filename)
            writer.append_data(image)
    shutil.rmtree('temp/')
    return ax1, ax2

waypoints = read_file("../../Data/2D_waypoints.txt")

car1_states_1 = read_pickle("../../Data/MPC_vs_MPC/scenario4_1_car1_states.pickle")
car1_states_2 = read_pickle("../../Data/MPC_vs_MPC/scenario4_2_car1_states.pickle")
car1_states_3 = read_pickle("../../Data/MPC_vs_MPC/scenario4_3_car1_states.pickle")
car1_states_4 = read_pickle("../../Data/MPC_vs_MPC/scenario4_4_car1_states.pickle")
car1_states_5 = read_pickle("../../Data/MPC_vs_MPC/scenario4_5_car1_states.pickle")

car2_states_1 = read_pickle("../../Data/MPC_vs_MPC/scenario4_1_car2_states.pickle")
car2_states_2 = read_pickle("../../Data/MPC_vs_MPC/scenario4_2_car2_states.pickle")
car2_states_3 = read_pickle("../../Data/MPC_vs_MPC/scenario4_3_car2_states.pickle")
car2_states_4 = read_pickle("../../Data/MPC_vs_MPC/scenario4_4_car2_states.pickle")
car2_states_5 = read_pickle("../../Data/MPC_vs_MPC/scenario4_5_car2_states.pickle")

idx = np.amin([car1_states_1.shape[0], car1_states_2.shape[0], car1_states_3.shape[0], car1_states_4.shape[0], car1_states_5.shape[0]])

car1_states_all = np.dstack((car1_states_1[:idx], car1_states_2[:idx], car1_states_3[:idx], car1_states_4[:idx], car1_states_5[:idx]))
car2_states_all = np.dstack((car2_states_1[:idx], car2_states_2[:idx], car2_states_3[:idx], car2_states_4[:idx], car2_states_5[:idx]))

for i in range(5):
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1 = plot_track_boundaries(ax1, waypoints[80:350], 15)
    ax1, ax2 = plot_trajectory_velocity_evolution(ax1, ax2, car1_states_all[:, :, i], car2_states_all[:, :, i],
    ['Trajectory Plot - MPC v/s MPC', 'Velocity Plot - MPC v/s MPC'],
    ['Car 1', 'Car 2'], '../../GIFS/MPC_vs_MPC/scene4_{}_traj_vel.gif'.format(i+1))
    plt.close()

f, ax = plt.subplots()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
ax.set_aspect('equal', 'box')

ax = plot_trajectory_evolution(ax, car1_states_all, car2_states_all,
 'Trajectories Plot - MPC v/s MPC',  '../../GIFS/MPC_vs_MPC/scene4_all_traj.gif')
plt.close()
