#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# Stanley Controller for controlling car in Carla Simulator

# NOTE: This script requires Carla simulator running in the background with the concerned map loaded 

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import os, sys
import getopt,glob
import pickle
import argparse
from typing import Tuple, NoReturn
import numpy as np

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

    def __init__(self):
        """Initialize simulation environment
        """
        # Connect to client
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(2.0)

        # Get World
        self.world = self.client.get_world()

        # Get Map of the world
        self.map = self.world.get_map()
        
        # Get all the blueprints available in the world
        self.blueprint_library =  self.world.get_blueprint_library()
        
        # Initialize state dataclass list to log subsequent state information
        self.states = []

        # Initialize control dataclass list to log subsequent control commands
        self.controls = []

        # Data for longitudinal controller
        self.vel_control = [velocity_control_var(0.0, 0.0)]

        # Track errors
        self.errors = []


    def spawn_vehicle_2D(self, spawn_pose: np.ndarray) -> NoReturn:
        """Spawn a Vehicle at given 2D pose

        Arguments
        ---------
        - spawn_pose: Vehicle spawn pose [x, y, heading]
        """
        # Load Tesla Model 3 blueprint
        self.car_model = self.blueprint_library.find('vehicle.tesla.model3')

        # Spawn vehicle at given pose
        self.spawn_point = carla.Transform(carla.Location(spawn_pose[0], spawn_pose[1], 2), carla.Rotation(0, spawn_pose[2], 0))
        self.vehicle =  self.world.spawn_actor(self.car_model, self.spawn_point)

        # Get max steering angle of car's front wheels
        self.max_steer_angle = self.vehicle.get_physics_control().wheels[0].max_steer_angle
        
        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []
        # Append our vehicle to actor list
        self.actor_list.append(self.vehicle)

    def save_log(self, filename: str, data: object) -> NoReturn:
        """Logging data to a .pickle file

        Arguments
        ---------
        - filename: Name of the file to store data
        - data: Data to be logged
        """
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def destroy(self) -> NoReturn:
        """Destroy all actors in the world
        """
        for actor in self.actor_list:
            actor.destroy()

    def set_tuning_params(self, kp: float = 0.5, ki: float = 0.0, kd: float = 0.1, ke: float = 0.1, kv: float = 10.0) -> NoReturn:
        """Set the tuning parameters for longitudinal controller and lateral controller

        Arguments
        ---------
        - kp: Proportional Gain
        - ki: Integral Gain
        - kd: Differential Gain
        - ke: Lateral Tracking Gain
        - kv: Low Velocity Gain
        """
        # PID control gains
        self.kp = kp
        self.ki = ki        
        self.kd = kd

        # Crosstrack error control gains
        self.ke = ke
        self.kv = kv


    def longitudinal_controller(self, v: float, v_des: float, prev_err: float, cumulative_error: float, tuning_params: list, dt: float) -> Tuple[float, float, float]:
        """Compute control signal (acceleration/deceleration) for linear velocity control

        Arguments
        ---------
        - v: Current velocity
        - v_des: Velocity setpoint
        - prev_error: Velocity error from previous control loop iteration for derivative controller
        - cumulative_error: Accumulated error over all previous control loop iterations for integral controller
        - tuning_params: [kp, ki, kd] PID controller tuning parameters
        - dt: Controller time step

        Returns
        -------
        - acc: Acceleration/deceleration control signal
        - curr_err: Velocity error from the current control loop
        - cumulative_error: Accumulated error upto current control loop
        """
        # Extract PID tuning parameters
        [kp, ki, kd] = tuning_params

        # Compute error between current and desired value
        curr_err = v_des - v

        cumulative_error += (curr_err * dt)

        # Acceleration/Deceleration control signal
        acc = (kp * curr_err) + (kd * (curr_err - prev_err) / dt) + (ki * cumulative_error)

        return acc, curr_err, cumulative_error


    def calculate_target_index(self, x_r: float, y_r: float, xs_des: np.ndarray, ys_des: np.ndarray, lookahead_idx: int = 2) -> Tuple[int, float]:
        """Compute the waypoint index which is 'lookahead_idx' indices ahead of the closest waypoint to the current robot position

        Arguments
        ---------
        - x_r: Vehicle's x position in world frame
        - y_r: Vehicle's y position in world frame
        - xs_des: Desired trajectory x coordinates in world frame
        - ys_des: Desired trajectory y coordinates in world frame
        - lookahead_idx: Number of indices to lookahead from the nearest waypoint to the vehicle

        Returns
        -------
        - idx: waypoint index
        - d: distance between vehicle and waypoint at index 'idx'
        """
        # Search nearest waypoint index
        dx = [x_r - x for x in xs_des]
        dy = [y_r - y for y in ys_des]
        dist = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        idx = (np.argmin(dist) + lookahead_idx) % len(xs_des)
        d = dist[idx]

        return idx, d


    def stanley_control(self, waypoints: np.ndarray, laps_required: int) -> NoReturn:
        """Deploy Stanley Control Paradigm for vehicle control

        Arguments
        ---------
        - waypoints: Desired trajectory for vehicle (x, y, yaw)
        - laps_required: Number of laps to be completed
        """
        # Wait for car to stabilize after spawning
        while self.vehicle.get_velocity().z != 0:
            pass

        # Desired velocity [m/s]
        v_des = 14.0

        # Desired trajectory
        x_des = waypoints[:,0]
        y_des = waypoints[:,1]
        yaw_des = [wrapToPi(i) for i in waypoints[:, 2]] # [radians] 
        
        prev_idx = 0
        laps_completed = 0

        while 1:
            x = self.vehicle.get_transform().location.x
            y = self.vehicle.get_transform().location.y
            yaw = self.vehicle.get_transform().rotation.yaw     # [degrees]
                        
            v_lon = self.vehicle.get_velocity().x * np.cos(np.radians(yaw)) + self.vehicle.get_velocity().y * np.sin(np.radians(yaw))
            v_lat = -self.vehicle.get_velocity().x * np.sin(np.radians(yaw)) + self.vehicle.get_velocity().y * np.cos(np.radians(yaw))

            self.snapshot = self.world.wait_for_tick()
            curr_t = self.snapshot.timestamp.elapsed_seconds # [seconds]
            
            # Velocity control
            dt = curr_t - self.states[-1].time

            # Append state
            self.states.append(state(curr_t, x, y, yaw, v_lon, v_lat, laps_completed))

            # Longitudinal Controller
            acc, v_error, acc_error = self.longitudinal_controller(
                v_lon, v_des, self.vel_control[-1].prev_err, self.vel_control[-1].acc_error, [self.kp, self.ki, self.kd], dt)

            # Append longitudinal controller error
            self.vel_control.append(velocity_control_var(v_error, acc_error))

            # Find nearest waypoint
            idx, d = self.calculate_target_index(x, y, x_des, y_des, 3)

            # Stop at end of track
            if (idx == waypoints.shape[0] - 1) and idx != prev_idx:
                laps_completed += 1
                if laps_completed == laps_required:
                    while v_lon != 0.0:
                        self.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0, reverse = False, brake = 0.2))
                        v_lon = self.vehicle.get_velocity().x * np.cos(np.radians(yaw)) + self.vehicle.get_velocity().y * np.sin(np.radians(yaw))
                        v_lat = self.vehicle.get_velocity().x * np.sin(np.radians(yaw)) - self.vehicle.get_velocity().y * np.cos(np.radians(yaw))
                    break
            
            prev_idx = idx

            if prev_idx == 1000:
                v_des = 6
                print('changing speed to {} m/s'.format(v_des))

            if prev_idx == 2000:
                v_des = 12
                print('changing speed to {} m/s'.format(v_des))

            # Heading error [radians]
            psi_h = wrapToPi(yaw_des[idx] - np.radians(yaw))

            # Crosstrack yaw difference to path yaw [radians]
            yaw_diff = wrapToPi(yaw_des[idx] - np.arctan2(y - y_des[idx], x - x_des[idx]))
        
            # Crosstrack error in yaw [radians]
            psi_c = np.arctan2(self.ke * np.sign(yaw_diff) * d, self.kv + v_lon)

            self.errors.append(track_error(curr_t, psi_h, psi_c, laps_completed))

            # Steering angle control
            _steer = np.degrees(wrapToPi(psi_h + psi_c))  # uncontrained in degrees
            _steer = max(min(_steer, self.max_steer_angle), -self.max_steer_angle)
            _steer = (_steer)/self.max_steer_angle # constrained to [-1, 1]

            # Split velocity control into throttle and brake and constrain them to [0, 1]
            if acc >= 0:
                _throttle = np.tanh(acc)
                _brake = 0
                if (_throttle - self.controls[-1].throttle) > 0.1:
                    _throttle = self.controls[-1].throttle + 0.1
            else:
                _throttle = 0
                _brake = np.tanh(abs(acc))
                if (_brake - self.controls[-1].brake) > 0.1:
                    _brake = self.controls[-1].brake + 0.1
            
            # Append control data 
            self.controls.append(control(curr_t, _throttle, _brake, _throttle - _brake, _steer * self.max_steer_angle, laps_completed))
            
            # Apply control
            self.vehicle.apply_control(carla.VehicleControl(
                throttle = _throttle, steer = _steer, reverse = False, brake = _brake))

def main():
    try:
        # Initialize car environment
        env = CarEnv()

        # Load waypoints
        waypoints = env.read_file("../../Data/2D_waypoints.txt")

        # Spawn a vehicle at spawn_pose
        spawn_pose = waypoints[0]
        spawn_pose[2] = np.degrees(spawn_pose[2])
        env.spawn_vehicle_2D(spawn_pose)

        # Set controller tuning params

        # Read command line args
        argparser = argparse.ArgumentParser(description = __doc__)
        argparser.add_argument('-e', '--ke', default=0.1, type=float, help='Lateral Tracking Gain for lateral control')
        argparser.add_argument('-v', '--kv', default=5.0, type=float, help='Low Velocity Gain for lateral control')
        argparser.add_argument('-p', '--kp', default=0.5, type=float, help='Proportional Gain for longitudinal control')
        argparser.add_argument('-i', '--ki', default=0.01, type=float, help='Integral Gain for longitudinal control')
        argparser.add_argument('-d', '--kd', default=0.15, type=float, help='Differential Gain for longitudinal control')
        argparser.add_argument('-n', '--number-of-laps', default=1, type=int, help='number of laps desired in the race')
        argparser.add_argument('-s', '--save', action='store_true', help='Set to True to save the states and control data')
        args = argparser.parse_args()

        env.set_tuning_params(args.kp, args.ki, args.kd, args.ke, args.kv)

        # Append initial state and controls
        curr_t = env.world.wait_for_tick().timestamp.elapsed_seconds
        env.states.append(state(curr_t, spawn_pose[0], spawn_pose[1], spawn_pose[2], 0.0, 0.0, 0))
        env.controls.append(control(curr_t, 0.0, 0.0, 0.0, 0.0, 0))

        # Initialize control loop
        env.stanley_control(waypoints, args.number_of_laps)
    
    finally:
        if args.save:
        # Save all Dataclass object lists to a file
            env.save_log('../../Data/states_e(%f)_v(%f)_p(%f)_i(%f)_d(%f)_n(%f).pickle'%(env.ke, env.kv, env.kp, env.ki, env.kd, args.number_of_laps), env.states)
            env.save_log('../../Data/controls_e(%f)_v(%f)_p(%f)_i(%f)_d(%f)_n(%f).pickle'%(env.ke, env.kv, env.kp, env.ki, env.kd, args.number_of_laps), env.controls)
            env.save_log('../../Data/errors_e(%f)_v(%f)_p(%f)_i(%f)_d(%f)_n(%f).pickle'%(env.ke, env.kv, env.kp, env.ki, env.kd, args.number_of_laps), env.errors)

        # Destroy all actors in the simulation
        env.destroy()

if __name__=="__main__":
    main()