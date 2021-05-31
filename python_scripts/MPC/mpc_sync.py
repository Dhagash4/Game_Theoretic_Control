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
from scipy.misc import derivative
from scipy import interpolate as interp

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

        # Set Synchronous Mode
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        # Get Map of the world
        self.map = self.world.get_map()
        
        # Get all the blueprints available in the world
        self.blueprint_library =  self.world.get_blueprint_library()
        
        # Initialize state dataclass list to log subsequent state information
        self.states = []

        # Initialize control dataclass list to log subsequent control commands
        self.controls = []

        # Track errors
        self.errors = []

        # Waypoints
        self.waypoints = []

        # Nearest Waypoints Index
        self.nearest_wp_idx = 0

        # Vehicle Spawn Index
        self.spawn_idx = 0

        # Vehicle Current Pose
        self.car_pose = []

        # Control Command
        self.control = np.array([0, 0])

    def spawn_vehicle_2D(self):
        """Spawn a Vehicle at given 2D pose

        Arg:
            spawn_pose: Vehicle spawn pose [x, y, heading]
        """
        # Load Tesla Model 3 blueprint
        self.car_model = self.blueprint_library.find('vehicle.tesla.model3')

        # Spawn vehicle at given pose
        self.spawn_pose = self.waypoints[self.spawn_idx]
        self.spawn_tf = carla.Transform(carla.Location(self.spawn_pose[0], self.spawn_pose[1], 2), carla.Rotation(0, np.degrees(self.spawn_pose[2]), 0))
        self.vehicle =  self.world.spawn_actor(self.car_model, self.spawn_tf)

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
        data = np.loadtxt(path, delimiter=delimiter)
        return data


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
        # End Synchronous Mode
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)

        for actor in self.actor_list:
            actor.destroy()


    def get_true_state(self):
        """Get vehicles true state from the simulation
        
        Returns:
            - true_state: True state of the vehicle [x, y, yaw, vx, vy]
        """
        x = self.vehicle.get_transform().location.x
        y = self.vehicle.get_transform().location.y
        yaw = wrapToPi(np.radians(self.vehicle.get_transform().rotation.yaw))     # [radians]
        vx = self.vehicle.get_velocity().x * np.cos(yaw) + self.vehicle.get_velocity().y * np.sin(yaw)
        vy = -self.vehicle.get_velocity().x * np.sin(yaw) + self.vehicle.get_velocity().y * np.cos(yaw)

        self.car_pose = np.array([x, y, yaw, vx, vy])


    def calculate_nearest_index(self):
        # Search nearest waypoint index
        x_r, y_r = self.car_pose[:2]
        fromIdx = max(0, self.nearest_wp_idx - 10)
        toIdx = self.nearest_wp_idx + 10

        dx = np.array([(x_r - x) ** 2 for x in self.waypoints[fromIdx:toIdx, 0]])
        dy = np.array([(y_r - y) ** 2 for y in self.waypoints[fromIdx:toIdx, 1]])
        dist = dx + dy

        self.nearest_wp_idx = (np.argmin(dist) + fromIdx)# % self.waypoints.shape[0]

    def calculate_error(self, state_mpc):
        x = state_mpc[0]
        y = state_mpc[1]
        vx = state_mpc[3]

        self.l = self.l + (vx * self.dt)

        x_ref = self.lut_x(self.l)
        y_ref = self.lut_y(self.l)
        yaw = self.lut_theta(self.l)

        e_c = sin(yaw) * (x - x_ref) - cos(yaw) * (y - y_ref)
        e_l = -cos(yaw) * (x - x_ref) - sin(yaw) * (y - y_ref)

        return MX(vcat([e_c, e_l]))


    def set_mpc_params(self, P, C, lookahead, vx_des):
        self.P = P
        self.C = C
        self.lookahead = lookahead
        self.vx_des = vx_des
        
    def set_opti_weights(self, w_u0, w_u1, w_e0, w_e1, w_vx, w_c):
        self.w_u0 = w_u0
        self.w_u1 = w_u1
        self.w_e0 = w_e0
        self.w_e1 = w_e1
        self.w_vx = w_vx
        self.w_c = w_c

    def fit_curve(self):
        self.L = np.arange(0, 100)
        fromIdx = max(0, self.nearest_wp_idx - 10)
        toIdx = fromIdx + 100

        self.lut_x = interpolant('LUT', 'bspline', [self.L], self.waypoints[fromIdx:toIdx, 0])
        self.lut_y = interpolant('LUT', 'bspline', [self.L], self.waypoints[fromIdx:toIdx, 1])
        self.lut_theta = interpolant('LUT', 'bspline', [self.L], self.waypoints[fromIdx:toIdx, 2])

    def mpc(self):
        """Deploy Stanley Control Paradigm for vehicle control

        Args:
            - waypoints: Desired trajectory for vehicle (x, y, yaw)
            - laps_required: Number of laps to be completed
        """
        s = MX.sym('s', 5)
        x, y, yaw, vx, vy = s[0], s[1], s[2], s[3], s[4]

        u = MX.sym('u', 2)
        acc, delta = [u[0], u[1]]

        dt = MX.sym('dt')

        prediction = vertcat(x + sqrt((vx + 0.001) ** 2 + (vy + 0.001)** 2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                            y + sqrt((vx + 0.001) ** 2 + (vy + 0.001) ** 2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                            yaw + (sqrt((vx + 0.001) ** 2 + (vy + 0.001) ** 2) * tan(delta) * dt / sqrt((19.8025) + (4.95 * tan(delta) ** 2))),
                            vx + ((4.22 * acc) - (-0.0013 * sqrt((vx + 0.001) ** 2 + (vy + 0.001) ** 2) * vx - 0.362 * vx)) * dt,
                            vy - (1.318 * (atan2(vy, vx+0.001) - delta) + (-0.0013 * sqrt((vx + 0.001) ** 2 + (vy + 0.001) ** 2) + 0.362) * vy) * dt)
    
        pred = Function('pred', [s, u, dt], [prediction])

        opti = Opti()

        s = opti.variable(5, self.P + 1)
        e = opti.variable(2, self.P + 1)
        u = opti.variable(2, self.C)
        p = opti.parameter(5, 1)

        opti.minimize(self.w_e0 * sumsqr(e[0, :]) + self.w_e1 * sumsqr(e[1, :])
                     + self.w_vx * sumsqr(self.vx_des - s[3, :])
                     + self.w_u0 * sumsqr(u[0, :]) + self.w_u1 * sumsqr(u[1, :])
                     + self.w_c * sumsqr(u[:, :self.C-1] - u[:, 1:self.C])
                     )
        
        # Set using true states
        opti.set_value(p, self.car_pose)
        opti.subject_to(s[:, 0] == p)

        # print('read true state')
        opti.subject_to(opti.bounded(-1.0, u[0, :], 1.0))
        opti.subject_to(opti.bounded(-1.22, u[1, :], 1.22))

        for i in range(self.C):
            opti.subject_to(s[:, i+1] == pred(s[:, i], u[:, i], self.dt))
            # opti.subject_to(e[:, i] == self.err_to_tangent(s[:, i]))
            opti.subject_to(e[:, i] == self.calculate_error(s[:, i]))

        for k in range(self.C, self.P):
            opti.subject_to(s[:, k+1] == pred(s[:, k], u[:, self.C - 1], self.dt))
            # opti.subject_to(e[:, k] == self.err_to_tangent(s[:, k]))
            opti.subject_to(e[:, k] == self.calculate_error(s[:, k]))

        opti.subject_to(e[:, -1] == self.calculate_error(s[:, -1]))

        # Good Initialization
        opti.set_initial(s, np.vstack([self.car_pose] * (self.P + 1)).T)
        opti.set_initial(u, 0)# np.vstack([self.control] * self.C).T)

        p_opts = {"print_time": False, 'ipopt.print_level': 0}
        opti.solver('ipopt', p_opts)

        sol = opti.solve()

        # print("solution found")
        self.control = sol.value(u)[:, 0]

        # print('states: \n', sol.value(s))
        print('errors: \n', sol.value(e)[:, 0])
        print('controls: \n', sol.value(u)[:, 0])

def main():
    try:
        # Data logging flag
        save_flag = False

        # Initialize car environment
        env = CarEnv()

        # Load waypoints
        env.waypoints = env.read_file("../../Data/2D_waypoints.txt")

        # Spawn a vehicle at spawn_pose
        env.spawn_idx = 0
        env.nearest_wp_idx = env.spawn_idx
        env.spawn_vehicle_2D()

        # Set controller tuning params
        # Default params
        num_of_laps = 1
        prev_idx = 0
        laps_completed = 0

        # Read command line args
        argv = sys.argv[1:]
        if len(argv) != 0:
            opts, args = getopt.getopt(argv, shortopts='n:s:', longopts=['nlaps', 'save'])
            for tup in opts:
                if tup[0] == '-n':
                    num_of_laps = int(tup[1])
                elif tup[0] == '-s':
                    save_flag = True

        print(num_of_laps)
        # Append initial state and controls
        # curr_t = env.world.wait_for_tick().timestamp.elapsed_seconds # [seconds]
        env.states.append(state(0, env.spawn_pose[0], env.spawn_pose[1], env.spawn_pose[2], 0.0, 0.0, 0))
        env.controls.append(control(0, 0.0, 0.0, 0.0, 0.0, 0))

        # Initialize control loop
        env.set_mpc_params(P = 6, C = 4, lookahead = 5, vx_des = 7)
        env.set_opti_weights(w_u0 = 1, w_u1 = 1, w_e0 = 10, w_e1 = 1, w_vx = 5, w_c = 2)

        while(1):
            env.world.tick()
            env.get_true_state()
            env.calculate_nearest_index()
            env.fit_curve()
            env.dt = 0.5
            env.l = 0.0
            env.mpc()

            if env.control[0] > 0:
                throttle = env.control[0]
                brake = 0
            else:
                brake = env.control[0]
                throttle = 0

            steer = env.control[1] / 1.22

            env.vehicle.apply_control(carla.VehicleControl(
                throttle = throttle, steer = steer, reverse = False, brake = brake))

            if (env.nearest_wp_idx == env.waypoints.shape[0] - 1) and env.nearest_wp_idx != prev_idx:
                laps_completed += 1
                if laps_completed == num_of_laps:
                    while env.car_pose[3] != 0.0:
                        env.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0, reverse = False, brake = 0.2))
                        env.get_true_state()
                        env.world.tick()
                    break

            prev_idx = env.nearest_wp_idx

    finally:
        if save_flag:
        # Save all Dataclass object lists to a file
            env.save_log('../../Data/states_mpc.pickle', env.states)
            env.save_log('../../Data/controls_mpc.pickle', env.controls)

        # Destroy all actors in the simulation
        env.destroy()


if __name__=="__main__":
    main()