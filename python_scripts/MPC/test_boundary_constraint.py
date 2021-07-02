#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

import glob
import getopt
import pickle
import os, sys
import numpy as np
from casadi import *
from typing import Tuple, NoReturn
from numpy.lib.utils import info
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.util import *
from Common.custom_dataclass import *

waypoints = np.loadtxt('../../Data/2D_waypoints.txt')

# plt.plot(waypoints[:, 0], waypoints[:, 1], 'r')

for waypoint in waypoints:
    x_ref = waypoint[0]
    y_ref = waypoint[1]
    yaw = waypoint[2]
    
    track_width = 15.0              # [m]
    d = (track_width * 0.8)/2
    
    a = -np.tan(yaw)
    b = 1
    c = (np.tan(yaw) * x_ref) - y_ref

    c1 = c - (d * np.sqrt(1 + (np.tan(yaw) ** 2)))
    c2 = c + (d * np.sqrt(1 + (np.tan(yaw) ** 2)))

    x_u = x_ref + d * np.sin(yaw)
    y_u = y_ref - d * np.cos(yaw)

    x_l = x_ref - d * np.sin(yaw)
    y_l = y_ref + d * np.cos(yaw)

    # u_b = a * x  + b * y + c1
    # l_b = a * x  + b * y + c2 
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'r')
    plt.axline((x_u, y_u), slope=np.tan(yaw), color='g')
    plt.axline((x_l, y_l), slope=np.tan(yaw), color='b')
    plt.pause(0.01)
    plt.clf()
