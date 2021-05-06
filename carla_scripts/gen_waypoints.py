#!/usr/bin/env python

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import time
import logging
import math
import random

import numpy as np


def main():
    try:
        # Connect to client
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Get World
        world = client.get_world()

        # Get Map of the world
        my_map = world.get_map()

        # Get waypoints for all road ids in the given map
        print("Extracting waypoints in the map....")
        waypoint_list_20 = my_map.get_waypoint_xodr(20, -1, 1).next_until_lane_end(1.0) 
        waypoint_list_21 = my_map.get_waypoint_xodr(21, -1, 1).next_until_lane_end(1.0) 
        print("{} Waypoints extracted successfully".format(len(waypoint_list_20) + len(waypoint_list_21)))

        time.sleep(5)

        # Iterate over all waypoints, visualize them and store their 2D pose
        x = []
        y = []
        yaw = []

        print('Visualizing all waypoints....')

        for waypoint in waypoint_list_20:
            world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                   color=carla.Color(r=0, g=255, b=0), life_time=100,
                                   persistent_lines=True)
            x.append(waypoint.transform.location.x)
            y.append(waypoint.transform.location.y)
            yaw.append(waypoint.transform.rotation.yaw)

        for waypoint in waypoint_list_21:
            world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                   color=carla.Color(r=255, g=255, b=0), life_time=100,
                                   persistent_lines=True)
            x.append(waypoint.transform.location.x)
            y.append(waypoint.transform.location.y)
            yaw.append(waypoint.transform.rotation.yaw)
        
        # Stack x, y and yaw[degrees] in a single matrix and save it as ASCII file
        waypoints = np.vstack((x, y, yaw))
        np.savetxt('2D_waypoints.txt', waypoints.T)
        print('2D waypoints saved in \'2D_waypoints.txt\'')

    finally:
        print('done')

if __name__=='__main__':
    main()