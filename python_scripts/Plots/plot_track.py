import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.plots import *
from Common.util import read_file

waypoints = read_file("../../Data/2D_waypoints.txt")

ax = plot_track_boundaries(waypoints)

plt.show()
