import os
import sys
import glob
import time
import logging
import numpy as np
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

from util import *

#Import CARLA anywhere on the system
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

@dataclass
class state():
    def __init__(self, time, pose_x, pose_y, pose_yaw, velocity):
        self.time = time
        self.pose_x = pose_x
        self.pose_y = pose_y
        self.pose_yaw = pose_yaw
        self.velocity = velocity

@dataclass
class control():
    def __init__(self, time, throttle, brake, steer):
        self.time = time
        self.throttle = throttle
        self.brake = brake
        self.steer = steer

@dataclass
class track_error():
    def __init__(self, heading_error, crosstrack_error):
        self.heading_error = heading_error
        self.crosstrack_error = crosstrack_error

@dataclass
class velocity_control_var():
    def __init__(self, prev_error, acc_error):
        self.prev_err = prev_error
        self.acc_error = acc_error


class CarEnv():

    def __init__(self, spawn_pose: np.ndarray):
        '''
        Initialize car environment
        '''
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


    def spawn_vehicle_2D(self, spawn_pose):
        '''
        Spawn a Vehicle at given 2D pose
        Arg:
            spawn_pose: Vehicle spawn pose [x, y, heading]
        '''
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


    def read_waypoints_file(self, path):
        ''' Read waypoints list
        '''
        self.waypoints = np.loadtxt(path)
        return self.waypoints


    def save_log(self, filename: str, data: object):
        """
        Logging data
        Args:
        filename: name of the file to store data
        data: data to be logged
        """
        with open(filename, "wb") as f:
            pickle.dump(data_my, f)


    def destroy(self):
        ''' 
        Destroy all actors in the world
        '''
        for actor in self.actor_list:
            actor.destroy()


    def longitudinal_controller(self, v: float, v_des: float, prev_err: float, cumulative_error: float, tuning_params: list, dt: float) -> (float, float, float):
        """
        Compute control signal (acceleration/deceleration) for linear velocity control
        Args:
            - v: current velocity
            - v_des: velocity setpoint
            - prev_error: velocity error from previous control loop iteration for derivative controller
            - cumulative_error: accumulated error over all previous control loop iterations for integral controller
            - tuning_params: [kp, ki, kd] PID controller tuning parameters
            - dt: Controller time step
        Returns:
            - acc: acceleration/deceleration control signal
            - curr_err: velocity error from the current control loop
            - cumulative_error: accumulated error upto current control loop
        """
        # Extract PID tuning parameters
        [kp, ki, kd] = tuning_params

        # Compute error between current and desired value
        curr_err = v_des - v

        cumulative_error += (curr_err * dt)

        # Acceleration/Deceleration control signal
        acc = (kp * curr_err) + (kd * (curr_err - prev_err) / dt) + (ki * cumulative_error)

        return acc, curr_err, cumulative_error


    def calculate_target_index(self, x_r: float, y_r: float, xs_des: np.ndarray, ys_des: np.ndarray, lookahead_idx: Optional[int] = 2) -> (int, float):
        """
        Compute the waypoint index which is 'lookahead_idx' indices ahead of the closest waypoint to the current robot position
        Args:
            - x_r: vehicle's x position in world frame
            - y_r: vehicle's y position in world frame
            - xs_des: desired trajectory x coordinates in world frame
            - ys_des: desired trajectory y coordinates in world frame
            - lookahead_idx: number of indices to lookahead from the nearest waypoint to the vehicle

        Returns:
            - idx: waypoint index
            - d: distance between vehicle and waypoint at index 'idx'
        """
        # Search nearest waypoint index
        dx = [x_r - x for x in xs_des]
        dy = [y_r - y for y in ys_des]
        dist = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        min_ind = (np.argmin(dist) + lookahead_idx) % len(xs_des)
        min_dist = dist[min_ind]

        return idx, d

    def stanley_control(self, waypoints):
        #Applying Control to the Car

        # Desired velocity [m/s]
        v_des = 14.0

        # Desired trajectory
        x_des = waypoints[:,0]
        y_des = waypoints[:,1]
        yaw_des = [wrapToPi(np.radians(i)) for i in waypoints[:, 2]] # [radians] 
        
        # Initial velocity error
        err_v = v_des - np.sqrt((self.vehicle.get_velocity().x) ** 2 + (self.vehicle.get_velocity().y) ** 2)

        # Crosstrack error control gains
        # Smooth recovery
        self.ke = 0.1
        # Low velocity gain
        self.kv = 10

        # PID tuning parameters
        kp_lon, kd_lon, ki_lon = 0.5, 0.1, 0.0

        self.track_idx = []
        self.track_d = []
                
        self.track_x_des = []        
        self.track_y_des = []
        self.track_yaw_des = []

        i = 0

        endpoint_reached = False
        while not endpoint_reached:
            i += 1

            x = self.vehicle.get_transform().location.x
            y = self.vehicle.get_transform().location.y
            yaw = self.vehicle.get_transform().rotation.yaw     # [degrees]

            v = np.sqrt((self.vehicle.get_velocity().x) ** 2 + (self.vehicle.get_velocity().y) ** 2)

            self.snapshot = self.world.wait_for_tick()
            curr_t = self.snapshot.timestamp.elapsed_seconds # [seconds]
            
            self.states.append(state(curr_t, x, y, yaw, v))

            # Velocity control
            dt = curr_t - self.states[-1].time

            acc, v_error, acc_error = self.longitudinal_controller(
                v, v_des, self.vel_control[-1].prev_err, self.vel_control[-1].acc_error, [kp_lon, ki_lon, kd_lon], dt)

            self.vel_control.append(velocity_control_var(v_error, acc_error))

            # Find nearest waypoint
            idx, d = self.calculate_target_index(x, y, x_des, y_des, 3)

            # Stop at end of track
            if idx == waypoints.shape[0]:
                endpoint_reached = True
                v_des = 0.0
                while v != 0.0:
                    self.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0, reverse = False, brake = 0.25))
                    v = np.sqrt((self.vehicle.get_velocity().x) ** 2 + (self.vehicle.get_velocity().y) ** 2)
                break

            # Visualize waypoints
            self.world.debug.draw_string(carla.Location(waypoints[idx, 0], waypoints[idx, 1], 2), '.', draw_shadow=False,
                                   color=carla.Color(r=255, g=0, b=0), life_time=5,
                                   persistent_lines=True)

            # Heading error [radians]
            psi_h = wrapToPi(yaw_des[idx] - np.radians(yaw))

            # Crosstrack yaw difference to path yaw [radians]
            yaw_diff = wrapToPi(yaw_des[idx] - np.arctan2(y - y_des[idx], x - x_des[idx]))
        
            # Crosstrack error in yaw [radians]
            psi_c = np.arctan2(self.ke * np.sign(yaw_diff) * d, self.kv + v)

            self.errors.append(track_error(psi_h, psi_c))

            # Steering angle control
            _steer = np.degrees(wrapToPi(psi_h + psi_c))  # uncontrained in degrees
            _steer = max(min(_steer, self.max_steer_angle), -self.max_steer_angle)
            _steer = (_steer)/self.max_steer_angle # constrained to [-1, 1]
            self.track_steer.append(_steer)

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

            self.controls.append(control(curr_t, _throttle, _brake, _steer))
            
            # Apply control
            self.vehicle.apply_control(carla.VehicleControl(
                throttle = _throttle, steer = _steer, reverse = False, brake = _brake))
            
        
def main():
    try:
        # Initialize car environment
        env = CarEnv(spawn_pose)

        # Load waypoints
        waypoints = env.read_waypoints_file("2D_waypoints.txt")

        # Spawn a vehicle at spawn_pose
        spawn_pose = waypoints[0]
        env.spawn_vehicle_2D(spawn_pose)

        # Initialize control loop
        env.stanley_control(waypoints)
    
    finally:
        # Save all Dataclass object lists to a file
        env.save_log('states.pickle', self.states)
        env.save_log('controls.pickle', self.controls)
        env.save_log('errors.pickle', self.errors)

        env.destroy()


if __name__=="__main__":
    main()