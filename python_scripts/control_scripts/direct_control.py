#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import glob
import getopt
import pickle
import os, sys
import casadi
import numpy as np
from casadi import *
from typing import Tuple, NoReturn
from numpy.lib.utils import info
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.util import *
from Common.custom_dataclass import *

#Import CARLA anywhere on the system
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class CarEnv():

    def __init__(self) -> NoReturn:
        """Initialize simulation environment
        """
        # Connect to client
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(2.0)

        # Get World
        self.world = self.client.get_world()

        # Set Synchronous Mode
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(self.settings)

        # Get Map of the world
        self.map = self.world.get_map()
        
        # Get all the blueprints available in the world
        self.blueprint_library =  self.world.get_blueprint_library()

        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []

    def spawn_vehicle_2D(self, spawn_idx: int, waypoints: np.ndarray, offset: float) -> Tuple[carla.libcarla.Vehicle, np.ndarray, float]:
        """Spawn a Vehicle at given index among the list of waypoints

        Arguments
        ---------
        - spawn_idx: spawn index from the waypoints list
        - waypoints: pre-computed waypoints to track
        - offset: offset distance (signed) from centerline

        Returns
        -------
        - vehicle: carla instance to the spawned vehicle
        - spawn_state: 5 parameter state of the vehicle [x, y, yaw, vx, vy]
        - max_steer_angle: maximum steering angle of the spawned vehicle [degrees]
        """
        # Load Tesla Model 3 blueprint
        car_model = self.blueprint_library.find('vehicle.tesla.model3')

        # Spawn vehicle at given pose
        spawn_state = np.r_[waypoints[spawn_idx], 0, 0]
        spawn_state[0] -= offset * np.sin(spawn_state[2])
        spawn_state[1] += offset * np.cos(spawn_state[2])

        spawn_tf = carla.Transform(carla.Location(spawn_state[0], spawn_state[1], 2), carla.Rotation(0, np.degrees(spawn_state[2]), 0))
        vehicle =  self.world.spawn_actor(car_model, spawn_tf)

        # Get max steering angle of car's front wheels
        max_steer_angle = vehicle.get_physics_control().wheels[0].max_steer_angle
        
        # Append our vehicle to actor list
        self.actor_list.append(vehicle)

        return vehicle, spawn_state, max_steer_angle

    def CameraSensor(self):
        #RGB Sensor 
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x','640')
        self.rgb_cam.set_attribute('image_size_y', '480')
        self.rgb_cam.set_attribute('fov', '110')
        
        #Attaching sensor to car
        transform = carla.Transform(carla.Location(x=-4.8,y=0.0, z=7.3), carla.Rotation(pitch = -30))
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle_1)
        self.actor_list.append(self.cam_sensor)
        self.cam_sensor.listen(lambda image: image.save_to_disk('/home/dhagash/MS-GE-02/MSR-Project/stanley_different_params/img%06d.png' % image.frame))

    def destroy(self) -> NoReturn:
        """Destroy all actors in the world
        """
        # End Synchronous Mode
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)

        for actor in self.actor_list:
            actor.destroy()
    
def main():
    try:
        # Initialize car environment
        env = CarEnv()

        # Load waypoints and car parameters obtained from System ID
        waypoints = read_file("../../Data/2D_waypoints.txt")

        # Spawn two vehicles at spawn_pose corresponding to spawn_idx index in waypoints list
        spawn_idx = 200
        vehicle_1, env.car_1_state, max_steer_angle_1 = env.spawn_vehicle_2D(spawn_idx, waypoints, -4)
        vehicle_2, env.car_2_state, max_steer_angle_2 = env.spawn_vehicle_2D(spawn_idx, waypoints, 4)

        # Spawn time car stabilization
        for i in range(50):
            env.world.tick()
        
        with open('../../Data/GTC/controls_1_mpc_two_car_diff_speed_ahead.pickle', 'rb') as f:
            controls_1 = pickle.load(f)
        
        with open('../../Data/GTC/controls_2_mpc_two_car_diff_speed_ahead.pickle', 'rb') as f:
            controls_2 = pickle.load(f)
        
        throttle_1 = np.array([c.throttle for c in controls_1])
        brake_1 = np.array([c.brake for c in controls_1])
        steer_1 = np.array([c.steer for c in controls_1])

        throttle_2 = np.array([c.throttle for c in controls_2])
        brake_2 = np.array([c.brake for c in controls_2])
        steer_2 = np.array([c.steer for c in controls_2])

        for i in range(len(throttle_1)):
            
            vehicle_1.apply_control(carla.VehicleControl(throttle = throttle_1[i], steer = steer_1[i], reverse = False, brake = brake_1[i]))
            vehicle_2.apply_control(carla.VehicleControl(throttle = throttle_2[i], steer = steer_2[i], reverse = False, brake = brake_2[i]))
            env.world.tick()

    finally:
        env.destroy()

if __name__=="__main__":
    main()

    