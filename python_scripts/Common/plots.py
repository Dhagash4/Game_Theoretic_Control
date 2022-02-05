#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# This script defines some helper functions to plot data

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
from typing import NoReturn
from matplotlib import pyplot as plt

def plot_track_boundaries(ax: plt.axes, centerline: np.ndarray, w: float) -> (plt.axes):
    '''
    Plot the race-track boundaries

    Arguments
    ---------
    - ax: handle to the desired matplotlib figure
    - centerline: centerline coordinates of the track
    - w: track width

    Returns
    -------
    - ax: handle to the edited matplotlib figure
    '''
    boundary_left = np.vstack((centerline[:, 0] - 0.5 * w * np.sin(centerline[:, 2]), 
                           centerline[:, 1] + 0.5 * w * np.cos(centerline[:, 2])))

    boundary_right = np.vstack((centerline[:, 0] + 0.5 * w * np.sin(centerline[:, 2]), 
                            centerline[:, 1] - 0.5 * w * np.cos(centerline[:, 2])))

    ax.plot(boundary_left[0], boundary_left[1], 'r')
    ax.plot(boundary_right[0], boundary_right[1], 'r')

    return ax