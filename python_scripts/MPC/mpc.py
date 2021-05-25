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
import numpy as np
from casadi import *
from numpy.lib.utils import info
from scipy.sparse import csc_matrix

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

def calculate_error(curr_state, ref_state, coeff, yaw_des):
    theta = ref_state[2] 
    x = cos(-theta) * (curr_state[0] - ref_state[0]) - sin(-theta) * (curr_state[1] - ref_state[1])
    y = sin(-theta) * (curr_state[0] - ref_state[0]) - cos(-theta) * (curr_state[1] - ref_state[1])
   
    yaw = curr_state[2]

    y_pred = coeff[0] * (x ** 2) + coeff[1] * x + coeff[2]
    crosstrack = y_pred - y

    head_err = yaw_des - yaw

    return MX(vcat([crosstrack, head_err]))


def calculate_nearest_index(curr_state, waypoints: np.ndarray, curr_idx):
    # Search nearest waypoint index
    dx = np.array([(curr_state[0] - x)**2 for x in waypoints[max(0, curr_idx - 10):curr_idx + 30, 0]])
    dy = np.array([(curr_state[1] - y)**2 for y in waypoints[max(0, curr_idx - 10):curr_idx + 30, 1]])
    i = range(max(0, curr_idx - 10), curr_idx + 30)
    dist = dx + dy

    idx = np.argmin(dist)

    return i[idx]


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
        self.spawn_point = carla.Transform(carla.Location(spawn_pose[0], spawn_pose[1], 2), carla.Rotation(0, np.degrees(spawn_pose[2]), 0))
        self.vehicle =  self.world.spawn_actor(self.car_model, self.spawn_point)

        # Get max steering angle of car's front wheels
        self.max_steer_angle = self.vehicle.get_physics_control().wheels[0].max_steer_angle
        
        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []
        # Append our vehicle to actor list
        self.actor_list.append(self.vehicle)


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


    def get_true_state(self):
        """Get vehicles true state from the simulation
        
        Returns:
            - true_state: True state of the vehicle [x, y, yaw, vx, vy]
        """
        x = self.vehicle.get_transform().location.x
        y = self.vehicle.get_transform().location.y
        yaw = np.radians(self.vehicle.get_transform().rotation.yaw)     # [radians]
        vx = self.vehicle.get_velocity().x * np.cos(yaw) + self.vehicle.get_velocity().y * np.sin(yaw)
        vy = -self.vehicle.get_velocity().x * np.sin(yaw) + self.vehicle.get_velocity().y * np.cos(yaw)

        true_state = np.array([x, y, yaw, vx, vy])

        return true_state


    def mpc(self, waypoints: np.ndarray, start_idx, laps_required: int):
        """Deploy Stanley Control Paradigm for vehicle control

        Args:
            - waypoints: Desired trajectory for vehicle (x, y, yaw)
            - laps_required: Number of laps to be completed
        """
        # Wait for car to stabilize after spawning
        while self.vehicle.get_velocity().z != 0:
            pass
        
        curr_idx = start_idx
        curr_state = self.get_true_state()

        # Desired velocity [m/s]
        vx_des = 5.0
        vy_des = 0.0

        s = MX.sym('s', 5)
        x, y, yaw, vx, vy = s[0], s[1], s[2], s[3], s[4]

        u = MX.sym('u', 2)
        acc, delta = [u[0], u[1]]

        dt = MX.sym('dt')   

        prediction = vertcat(x + sqrt(vx**2 + vy**2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                            y + sqrt(vx**2 + vy**2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                            atan2(sin(yaw + (sqrt(vx**2 + vy**2) * tan(delta) * dt / sqrt((19.8025) + (4.95 * tan(delta)**2)))), cos(yaw + (sqrt(vx**2 + vy**2) * tan(delta) * 0.03 / sqrt((19.8025) + (4.95 * tan(delta)**2))))),
                            vx + ((4.22 * acc) - (-0.0013 * sqrt(vx**2 + vy**2) * vx - 0.362 * vx)) * dt,
                            vy - (1.318 * (atan2(vy, vx) - delta) + (-0.0013 * sqrt(vx**2 + vy**2) + 0.362) * vy) * dt)
    
        pred = Function('pred', [s, u, dt], [prediction])

        opti = Opti()

        # Prediction Horizon
        i = 0
        while(i < 1000):
            i += 1
            # Set using true states
            curr_idx = calculate_nearest_index(curr_state, waypoints, curr_idx)
            yaw_des = waypoints[curr_idx, 2]
            yaw = -curr_state[2]
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            wp_car_frame = R @ (waypoints[max(0, curr_idx-10):curr_idx+10, :2].T - curr_state[:2].reshape(-1, 1))
            coeff = np.polyfit(wp_car_frame[0, :], wp_car_frame[1, :], 2)
            coeff_list = coeff.tolist()

            N = 5
            s = opti.variable(5, N + 1)
            e = opti.variable(2, N + 1)
            u = opti.variable(2, N)
            p = opti.parameter(5, 1)

            opti.minimize(sumsqr(e) + sumsqr(vx_des - s[3, :]) + sumsqr(u) + sumsqr(u[:, :N-1] - u[:, 1:N]))

            for k in range(N):
                opti.subject_to(e[:, k] == calculate_error(s[:, k], curr_state, coeff_list, yaw_des))
                opti.subject_to(s[:, k+1] == pred(s[:, k], u[:, k], 0.03))

            opti.subject_to(s[:, 0] == p)

            # Set using true states
            self.snapshot = self.world.wait_for_tick()
            curr_t = self.snapshot.timestamp.elapsed_seconds # [seconds]
            curr_state = self.get_true_state()
            self.states.append(state(curr_t, curr_state[0], curr_state[1], curr_state[2], curr_state[3], curr_state[4], 0))
            curr_idx = calculate_nearest_index(curr_state, waypoints, curr_idx)

            opti.set_value(p, curr_state)
            print('read true state')
            opti.subject_to(opti.bounded(-1, u[0, :], 1))
            opti.subject_to(opti.bounded(-1.22, u[1, :], 1.22))

            opti.set_value(p, curr_state)
            print('read true state')
            opti.subject_to(opti.bounded(-1, u[0, :], 1))
            opti.subject_to(opti.bounded(-1.22, u[1, :], 1.22))

            # Good Initialization
            opti.set_initial(s, np.array([curr_state, curr_state, curr_state, curr_state, curr_state, curr_state]).T)

            p_opts = {"print_time": True, 'ipopt.print_level': 0}
            opti.solver('ipopt', p_opts)

            sol = opti.solve()

            print("solution found")
            cont = sol.value(u)

            if cont[0, 0] > 0:
                throttle = cont[0, 0]
                brake = 0
            else:
                brake = cont[0, 0]
                throttle = 0

            steer = cont[1, 0] / 1.22

            print('states', sol.value(s))
            print('errors', sol.value(e))
            print('controls', sol.value(u))

            self.snapshot = self.world.wait_for_tick()
            curr_t = self.snapshot.timestamp.elapsed_seconds # [seconds]
            
            # Append control data 
            self.controls.append(control(curr_t, _throttle, _brake, _throttle - _brake, _steer * self.max_steer_angle, laps_completed))
            
            print('Apply control')
            # Apply control
            self.vehicle.apply_control(carla.VehicleControl(
                throttle = throttle, steer = steer, reverse = False, brake = brake))
            for i in range(100):
                pass

def main():
    try:
        # Data logging flag
        save_flag = False

        # Initialize car environment
        env = CarEnv()

        # Load waypoints
        waypoints = env.read_file("../../Data/2D_waypoints.txt")

        # Spawn a vehicle at spawn_pose
        spawn_idx = 1
        spawn_pose = waypoints[spawn_idx]
        env.spawn_vehicle_2D(spawn_pose)

        # Set controller tuning params
        # Default params
        num_of_laps = 1

        # Read command line args
        argv = sys.argv[1:]
        if len(argv) != 0:
            opts, args = getopt.getopt(argv, shortopts='n:s:', longopts=['nlaps', 'save'])
            for tup in opts:
                if tup[0] == '-n':
                    num_of_laps = int(tup[1])
                elif tup[0] == '-s':
                    save_flag = True

        # Append initial state and controls
        curr_t = env.world.wait_for_tick().timestamp.elapsed_seconds # [seconds]
        env.states.append(state(curr_t, spawn_pose[0], spawn_pose[1], spawn_pose[2], 0.0, 0.0, 0))
        env.controls.append(control(curr_t, 0.0, 0.0, 0.0, 0.0, 0))

        # Initialize control loop
        env.mpc(waypoints, spawn_idx, num_of_laps)
    
    finally:
        if save_flag:
        # Save all Dataclass object lists to a file
            env.save_log('../../Data/states_mpc.pickle', env.states)
            env.save_log('../../Data/controls_mpc.pickle', env.controls)

        # Destroy all actors in the simulation
        env.destroy()


if __name__=="__main__":
    main()