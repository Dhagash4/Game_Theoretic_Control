import glob
import os
import sys
import random
import time
import numpy as np

from stanley_control import *
from util import *
from datetime import datetime

#Import CARLA anywhere on the system
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarEnv():

    def __init__(self, waypoints):
        ''' Initialize car environment
        '''
        # Connect to client
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(2.0)

        # Get World
        self.world = self.client.get_world()

        # Get Map of the world
        self.map = world.get_map()

        # Get all the blueprints available in the world
        blueprint_library =  self.world.get_blueprint_library()

        # Load Tesla Model 3 blueprint
        self.car_model = blueprint_library.find('vehicle.tesla.model3')

        # Spawn vehicle at first waypoint
        self.spawn_point = carla.transform(carla.location(waypoints[0, 0], waypoints[0, 1], 2), carla.rotation(0, waypoints[0, 2], 0))
        self.vehicle =  self.world.spawn_actor(self.car_model, self.spawn_point)

        # Get max steering angle of car's front wheels
        self.max_steer_angle = self.vehicle.get_physics_control().wheels[0].max_steer_angle
        
        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []
        # Append our vehicle to actor list
        self.actor_list.append(self.vehicle)

        # Previous states
        self.prev_t = 0.0
        self.prev_v_error = 0.0
        self.prev_throttle = 0.0
        self.int_error_v = 0.0

    
    def destroy(self):
        ''' Destroy all actors in the world
        '''
        for actor in self.actor_list:
            actor.destroy()


    def longitudinal_controller(v: float, v_des: float, prev_err: float, cumulative_error: float, tuning_param: list, dt: float):
        """
        Compute control signal (acceleratio/deceleration) for linear velocity control
        Args:
            - v (float): current velocity
            - v_des (float): velocity setpoint
            - prev_error (float): velocity error from previous control loop iteration for derivative controller
            - cumulative_error (float): accumulated error over all previous control loop iterations for integral controller
            - tuning_param (1x3 array of floats): [kp, ki, kd]: PID controller tuning parameters
            - dt (float): Controller time step
        Returns:
            - acc (float): acceleration/deceleration control signal
            - curr_err (float): velocity error from the current control loop
            - cumulative_error (float): accumulated error upto current control loop
        """
        # Extract PID tuning parameters
        [kp, ki, kd] = tuning_param

        # Compute error between current and desired value
        curr_err = v_des - v

        cumulative_error += (curr_err * dt)

        # Acceleration/Deceleration control signal
        acc = (kp * curr_err) + (kd * (curr_err - prev_err) / dt) + (ki * cumulative_error)

        return acc, curr_err, cumulative_error


    def calculate_target_index(x_r, y_r, xs_des: list, ys_des: list()):
        # Search nearest waypoint index
        dx = [xr - x for x in xs_des]
        dy = [yr - y for y in ys_des]
        dist = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        min_ind = np.argmin(dist)
        min_dist = np.min(dist)

        return min_ind, min_dist


    def stanley_control(self, waypoints):
        # Saving the state before control
        acc_before = self.vehicle.get_acceleration()
        ang_vel_before = self.vehicle.get_angular_velocity()
        vel_before = self.vehicle.get_velocity()
        loc_before = self.vehicle.get_transform()

        #Applying Control to the Car
        
        # Desired velocity [m/s]
        v_des = 1.0

        # Desired trajectory
        x_des = waypoints[:,0]
        y_des = waypoints[:,1]
        yaw_des = [wrapToPi(np.radians(i)) for i in waypoints[:, 2]]
        
        # Initial velocity error
        err_v = v_des - self.vehicle.get_velocity().x
        err_total_v = 0

        # Crosstrack error control gains
        # Smooth recovery
        ke = 0.8
        # Low velocity gain
        kv = 0.5

        # PID tuning parameters
        kp_lon, kd_lon, ki_lon = 1.0, 0.0, 0.0

        while 1:
            x = self.vehicle.get_transform().location.x
            y = self.vehicle.get_transform().location.y
            yaw = self.vehicle.get_transform().rotation.yaw     # [degrees]
            v = self.vehicle.get_velocity().x
            curr_t = carla.timestamp.elapsed_seconds() # [seconds]

            # Velocity control
            dt = curr_t - self.prev_t
            acc, self.prev_v_error, self.int_error_v = longitudinal_controller(
                v, v_des, self.prev_v_error, self.int_error_v, [kp_lon, ki_lon, kd_lon], dt)

            # Find nearest waypoint
            idx, d = calculate_target_index(x, y, x_des, y_des)

            # Heading error [radians]
            psi_h = wrapToPi(np.radians(yaw_des[idx] - yaw))

            # Crosstrack yaw difference to path yaw [radians]
            yaw_diff = wrapToPi(np.radians(yaw_des[idx]) - np.arctan2(y - y_des[idx], x - x_des[idx]))

            # Crosstrack error in yaw [radians]
            psi_c = np.arctan2(ke * np.sign(yaw_diff) * d, kv + v)

            # Steering angle control
            _steer = np.degrees(wrapToPi(psi_h + psi_c))  # uncontrained in degrees
            _steer = max(min(_steer, self.max_steer_angle), -self.max_steer_angle) # constrained to [-1, 1]

            # Split velocity control into throttle and brake and constrain them to [0, 1]
            if acc >= 0:
                _throttle = np.tanh(acc)
                _brake = 0
            else:
                _throttle = 0
                _brake = np.tanh(abs(acc))
            
            # Apply control
            self.vehicle.apply_control(carla.VehicleControl(
                throttle = _throttle, steer = _steer, reverse = False, brake = _brake))

            self.prev_t = curr_t
            
        #After Control Information


def read_waypoints_file(path):
    ''' Read waypoints list
    '''
    waypoints = np.loadtxt(path)
    return waypoints

        
def main():
    try:
        # Load waypoints
        waypoints = read_waypoints_file("2D_waypoints.txt")
        # Initialize car environment
        env = CarEnv(waypoints)

        env.stanley_control(waypoints)
    
    finally:
        env.destroy()

if __name__=="__main__":
    main()