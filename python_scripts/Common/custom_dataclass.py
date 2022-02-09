#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# This script defines custom dataclasses to store data

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from dataclasses import dataclass

# Class to store states
@dataclass
class state():
    def __init__(self, time, pose_x, pose_y, pose_yaw, v_lon, v_lat, lap_num):
        self.time = time
        self.pose_x = pose_x
        self.pose_y = pose_y
        self.pose_yaw = pose_yaw
        self.v_lon = v_lon
        self.v_lat = v_lat
        self.lap_num = lap_num

# Class to store control commands
@dataclass
class control():
    def __init__(self, time, throttle, brake, acc, steer, lap_num):
        self.time = time
        self.throttle = throttle
        self.brake = brake
        self.acc = acc
        self.steer = steer
        self.lap_num = lap_num

# Class to store tracking errors
@dataclass
class track_error():
    def __init__(self, time, heading_error, crosstrack_error, lap_num):
        self.time = time
        self.heading_error = heading_error
        self.crosstrack_error = crosstrack_error
        self.lap_num = lap_num

# Class to store longitudinal controller error params
@dataclass
class velocity_control_var():
    def __init__(self, prev_error, acc_error):
        self.prev_err = prev_error
        self.acc_error = acc_error
