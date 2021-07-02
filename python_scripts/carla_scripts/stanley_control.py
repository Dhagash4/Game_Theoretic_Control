#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import os, sys
import getopt
import glob
import pickle
import numpy as np
from numpy.lib.utils import info

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


    def spawn_vehicle_2D(self, spawn_pose: np.ndarray):
        """Spawn a Vehicle at given 2D pose
        Arg:
            spawn_pose: Vehicle spawn pose [x, y, heading]
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

    
    def CameraSensor(self):

        #RGB Sensor 

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x','640')
        self.rgb_cam.set_attribute('image_size_y', '480')
        self.rgb_cam.set_attribute('fov', '110')
        

        #Attaching sensor to car

        transform = carla.Transform(carla.Location(x=-4.8,y=0.0, z=7.3), carla.Rotation(pitch = -30))
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.cam_sensor)
        self.cam_sensor.listen(lambda image: image.save_to_disk('/home/dhagash/MS-GE-02/MSR-Project/stanley_control/img%09d.png' % image.frame))


    def read_file(self, path: str, delimiter: str = ' ') -> (np.ndarray):
        """ Read data from a file
        Args:
            - path: Path of ASCII file to read
            - delimiter: Delimiter character for the file to be read
        
        Returns:
            - data: Data from file as a numpy array
        """
        self.data = np.loadtxt(path, delimiter=delimiter)
        return self.data


    def save_log(self, filename: str, data: object):
        """Logging data to a .pickle file
        Args:
            - filename: Name of the file to store data
            - data: Data to be logged
        """
        with open(filename, "wb") as f:
            pickle.dump(data, f)


    def destroy(self):
        """Destroy all actors in the world
        """
        for actor in self.actor_list:
            actor.destroy()


    def set_tuning_params(self, kp: float = 0.5, ki: float = 0.0, kd: float = 0.1, ke: float = 0.1, kv: float = 10.0):
        """Set the tuning parameters for longitudinal controller and lateral controller
        Args:
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


    def longitudinal_controller(self, v: float, v_des: float, prev_err: float, cumulative_error: float, tuning_params: list, dt: float):
        """Compute control signal (acceleration/deceleration) for linear velocity control
        Args:
            - v: Current velocity
            - v_des: Velocity setpoint
            - prev_error: Velocity error from previous control loop iteration for derivative controller
            - cumulative_error: Accumulated error over all previous control loop iterations for integral controller
            - tuning_params: [kp, ki, kd] PID controller tuning parameters
            - dt: Controller time step
        Returns:
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


    def calculate_target_index(self, x_r: float, y_r: float, xs_des: np.ndarray, ys_des: np.ndarray, lookahead_idx: int = 2):
        """Compute the waypoint index which is 'lookahead_idx' indices ahead of the closest waypoint to the current robot position
        Args:
            - x_r: Vehicle's x position in world frame
            - y_r: Vehicle's y position in world frame
            - xs_des: Desired trajectory x coordinates in world frame
            - ys_des: Desired trajectory y coordinates in world frame
            - lookahead_idx: Number of indices to lookahead from the nearest waypoint to the vehicle
        Returns:
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


    def stanley_control(self, waypoints: np.ndarray, laps_required: int):
        """Deploy Stanley Control Paradigm for vehicle control
        Args:
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
        
        # Initial velocity error
        err_v = v_des - np.sqrt((self.vehicle.get_velocity().x) ** 2 + (self.vehicle.get_velocity().y) ** 2)

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
                print('changing speed')
                v_des = 6

            if prev_idx == 2000:
                print('changing speed')
                v_des = 12

            # Visualize waypoints
            self.world.debug.draw_string(carla.Location(waypoints[idx, 0], waypoints[idx, 1], 2), '.', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=5,
                                persistent_lines=True)

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
        # Data logging flag
        save_flag = False

        # Initialize car environment
        env = CarEnv()

        # Load waypoints
        waypoints = env.read_file("../../Data/2D_waypoints.txt")

        # Spawn a vehicle at spawn_pose
        spawn_pose = waypoints[0]
        spawn_pose[2] = np.degrees(spawn_pose[2])
        env.spawn_vehicle_2D(spawn_pose)
        # env.CameraSensor()
        # Set controller tuning params
        # Default params
        num_of_laps = 1
        kp, ki, kd, ke, kv = [0.5, 0.01, 0.15, 0.1, 5.0]

        # Read command line args
        argv = sys.argv[1:]
        if len(argv) != 0:
            opts, args = getopt.getopt(argv, shortopts='e:v:p:i:d:n:s:', longopts=['ke', 'kv', 'kp', 'ki', 'kd', 'nlaps', 'save'])
            for tup in opts:
                if tup[0] == '-e':
                    ke = float(tup[1])                    
                elif tup[0] == '-v':
                    kv = float(tup[1])
                elif tup[0] == '-p':
                    kp = float(tup[1])
                elif tup[0] == '-i':
                    ki = float(tup[1])
                elif tup[0] == '-d':
                    kd = float(tup[1])
                elif tup[0] == '-n':
                    num_of_laps = int(tup[1])
                elif tup[0] == '-s':
                    save_flag = True

            env.set_tuning_params(kp, ki, kd, ke, kv)
        else:
            env.set_tuning_params()

        # Append initial state and controls
        curr_t = env.world.wait_for_tick().timestamp.elapsed_seconds # [seconds]
        env.states.append(state(curr_t, spawn_pose[0], spawn_pose[1], spawn_pose[2], 0.0, 0.0, 0))
        env.controls.append(control(curr_t, 0.0, 0.0, 0.0, 0.0, 0))

        # Initialize control loop
        env.stanley_control(waypoints, num_of_laps)
    
    finally:
        if save_flag:
        # Save all Dataclass object lists to a file
            env.save_log('../../Data/states_e(%f)_v(%f)_p(%f)_i(%f)_d(%f)_n(%f).pickle'%(env.ke, env.kv, env.kp, env.ki, env.kd, num_of_laps), env.states)
            env.save_log('../../Data/controls_e(%f)_v(%f)_p(%f)_i(%f)_d(%f)_n(%f).pickle'%(env.ke, env.kv, env.kp, env.ki, env.kd, num_of_laps), env.controls)
            env.save_log('../../Data/errors_e(%f)_v(%f)_p(%f)_i(%f)_d(%f)_n(%f).pickle'%(env.ke, env.kv, env.kp, env.ki, env.kd, num_of_laps), env.errors)

        # Destroy all actors in the simulation
        env.destroy()


if __name__=="__main__":
    main()