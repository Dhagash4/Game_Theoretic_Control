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

        # Nearest Waypoints Index
        self.nearest_wp_idx = 0

        # Vehicle Current Pose
        self.car_state = np.empty(5)

        # Control Command
        self.control = np.array([0, 0])

    def spawn_vehicle_2D(self, spawn_idx: int, waypoints: np.ndarray) -> (np.ndarray):
        """Spawn a Vehicle at given index among the list of waypoints

        Arguments
        ---------
        - spawn_idx: Vehicle spawn index from the waypoints list
        - waypoints: pre-computed waypoints to track

        Returns
        -------
        - spawn_state: 5 parameter state of the vehicle [x, y, yaw, vx, vy]
        """
        # Load Tesla Model 3 blueprint
        car_model = self.blueprint_library.find('vehicle.tesla.model3')

        # Spawn vehicle at given pose
        spawn_state = np.r_[waypoints[spawn_idx], 0, 0]
        spawn_tf = carla.Transform(carla.Location(spawn_state[0], spawn_state[1], 2), carla.Rotation(0, np.degrees(spawn_state[2]), 0))
        self.vehicle =  self.world.spawn_actor(car_model, spawn_tf)

        # Get max steering angle of car's front wheels
        self.max_steer_angle = self.vehicle.get_physics_control().wheels[0].max_steer_angle
        
        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []
        # Append our vehicle to actor list
        self.actor_list.append(self.vehicle)

        # Add spawn state to the states data log
        self.states.append(state(0, spawn_state[0], spawn_state[1], spawn_state[2], spawn_state[3], spawn_state[4], 0))

        return spawn_state


    def read_file(self, path: str, delimiter: str = ' ') -> (np.ndarray):
        """ Read data from a file

        Arguments
        ---------
        - path: Path of ASCII file to read
        - delimiter: Delimiter character for the file to be read
        
        Returns
        -------
        - data: Data from file as a numpy array
        """
        data = np.loadtxt(path, delimiter=delimiter)
        return data


    def save_log(self, filename: str, data: object) -> NoReturn:
        """Logging data to a '.pickle' file

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
        # End Synchronous Mode
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)

        for actor in self.actor_list:
            actor.destroy()


    def get_true_state(self) -> (np.ndarray):
        """Get vehicles true state from the simulation
        
        Returns
        -------
        - true_state: True state of the vehicle [x, y, yaw, vx, vy]
        """
        x = self.vehicle.get_transform().location.x
        y = self.vehicle.get_transform().location.y
        yaw = wrapToPi(np.radians(self.vehicle.get_transform().rotation.yaw))     # [radians]
        vx = self.vehicle.get_velocity().x * np.cos(yaw) + self.vehicle.get_velocity().y * np.sin(yaw)
        vy = -self.vehicle.get_velocity().x * np.sin(yaw) + self.vehicle.get_velocity().y * np.cos(yaw)

        true_state = np.array([x, y, yaw, vx, vy])

        return true_state

    def predict_new(self, old_state, control, params, dt):
        L, p, Cd, Cf, Cc = params

        x, y, theta, vx, vy = old_state
        v = np.sqrt(vx ** 2 + vy ** 2)

        acc, delta = control

        x_new = x + (v * np.cos(np.arctan2(np.tan(delta), 2) + theta) * dt)
        y_new = y + (v * np.sin(np.arctan2(np.tan(delta), 2) + theta) * dt)
        theta_new = wrapToPi(theta + (v * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
        vx_new = vx + (p * acc - Cd * v * vx - Cf * vx) * dt
        vy_new = vy - (Cc * wrapToPi(np.arctan2(vy, vx) - delta) + (Cd * v + Cf) * vy) * dt

        new_state = np.array([x_new, y_new, theta_new, vx_new, vy_new])

        return new_state

    def calculate_nearest_index(self, current_state: np.ndarray, waypoints: np.ndarray) -> (int):
        """Search nearest waypoint index to the current pose from the waypoints list

        Arguments
        ---------
        - current_state: current state of the vehicle
        - waypoints: pre-computed waypoints to track

        Returns
        -------
        - idx: index of the waypoint corresponding to minimum distance from current pose
        """
        # Extract vehicle position
        x_r, y_r = current_state[:2]
        # Compute squared distance to each waypoint
        dx = np.array([(x_r - x) ** 2 for x in waypoints[:, 0]])
        dy = np.array([(y_r - y) ** 2 for y in waypoints[:, 1]])
        dist = dx + dy
        # Find index corresponding to minimum distance
        idx = np.argmin(dist)

        return idx

    def calculate_error(self, state_mpc: MX, theta: MX) -> (MX):
        """Compute Contouring and Lag errors at given prediction step for the Model Predictive Contour Controller

        Arguments
        ---------
        - state_mpc: predicted state by the MPC loop (MX array [x, y, yaw, vx, vy])
        - theta: predicted path length parameter

        Returns
        -------
        - mpcc_error: MX array of contouring and lag error
        """
        x = state_mpc[0]
        y = state_mpc[1]

        x_ref = self.lut_x(theta)
        y_ref = self.lut_y(theta)
        yaw = self.lut_theta(theta)

        e_c = sin(yaw) * (x - x_ref) - cos(yaw) * (y - y_ref)
        e_l = -cos(yaw) * (x - x_ref) - sin(yaw) * (y - y_ref)

        mpcc_error = MX(vcat([e_c, e_l]))
        return mpcc_error

    def set_mpc_params(self, P: int, C: int, vmax: float) -> NoReturn:
        self.P = P
        self.C = C
        self.vmax = vmax
        self.theta = 0
        
    def set_opti_weights(self, w_u0: float, w_u1: float, w_qc: float, w_ql: float, gamma: float, w_c: float) -> NoReturn:
        self.w_u0 = w_u0
        self.w_u1 = w_u1
        self.w_qc = w_qc
        self.w_ql = w_ql
        self.gamma = gamma
        self.w_c = w_c

    def fit_curve(self, waypoints: np.ndarray) -> Tuple[Function, Function, Function]:
        """Fit a spline to the waypoints parametrized by the path length
        Arguments
        ---------
        - waypoints: pre-computed waypoints to fit a parametric spline to

        Returns
        -------
        - lut_x: Look up table for the x coordinate
        - lut_y: Look up table for the y coordinate
        - lut_theta: Look up table for the path orientation
        """
        L = np.arange(0, waypoints.shape[0])

        self.lut_x = interpolant('LUT_x', 'bspline', [L], waypoints[:, 0], dict(degree=[3]))
        self.lut_y = interpolant('LUT_y', 'bspline', [L], waypoints[:, 1], dict(degree=[3]))
        self.lut_theta = interpolant('LUT_t', 'bspline', [L], waypoints[:, 2], dict(degree=[1]))

        return self.lut_x, self.lut_y, self.lut_theta

    def mpc(self, prev_states, prev_controls, prev_t, prev_v, system_params: np.ndarray, max_L: float, Ts: float):
        """Model Predictive Controller Function

        Arguments
        ---------
        - system_params: Vehicle dynamics parameters obtained from System ID
        - max_L: maximum path length of the track
        - Ts: Sampling Time of the controller
        """        
        ##### Define state dynamics in terms of casadi function #####

        L, p, Cd, Cf, Cc = system_params
        d_nan = 1e-5 # Add small number to avoid NaN error at zero velocities in the Jacobian evaluation

        dt = MX.sym('dt')
        state = MX.sym('s', 5)
        control_command = MX.sym('u', 2)

        x, y, yaw, vx, vy = state[0], state[1], state[2], state[3], state[4]
        acc, delta = [control_command[0], control_command[1]]

        state_prediction = vertcat(x + sqrt((vx + d_nan) ** 2 + (vy + d_nan)** 2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                            y + sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                            yaw + (sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * tan(delta) * dt / sqrt((L ** 2) + (0.5 * L * tan(delta)) ** 2)),
                            vx + ((p * acc) - (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * vx - Cf * vx)) * dt,
                            vy - (Cc * (atan2(vy, vx + d_nan) - delta) + (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) + Cf) * vy) * dt)
    
        pred = Function('pred', [state, control_command, dt], [state_prediction])

        ##### Optistack begin #####
        opti = Opti()

        s = opti.variable(5, self.P + 1)    # States [x, y, yaw, vx, vy]
        e = opti.variable(2, self.P + 1)    # Errors [contouring, lag]
        t = opti.variable(1, self.P + 1)    # Path length parameter
        v = opti.variable(1, self.P + 1)    # Path progression rate
        u = opti.variable(2, self.C)        # Controls [throttle/brake, steering]

        # Costs to optimize over
        p_acc = self.w_u0 * sumsqr(u[0, :])                                     # Throttle/Brake cost
        p_steer = self.w_u1 * sumsqr(u[1, :])                                   # Steering cost
        p_control_roc = self.w_c * sumsqr(u[:, :self.C-1] - u[:, 1:self.C])     # Rate of Change of Controls
        p_vel = 0.0 * sumsqr(v)

        p_v_roc = self.w_c * sumsqr(v[:self.P] - v[1:self.P + 1])               # Rate of Change of progression rate 
        r_v_max = self.gamma * sum2(v) * Ts                                     # Progression Reward
        p_contouring = self.w_qc * sumsqr(e[0, :])                              # Contouring Error
        p_lag = self.w_ql * sumsqr(e[1, :])                                     # Lag Error
        
        # Minimization objective
        opti.minimize(p_contouring + p_lag + p_control_roc + p_v_roc + p_acc + p_steer
                     - r_v_max)
        
        # Constraints
        opti.subject_to(s[:, 0] == self.car_state)              # Start prediction with true vehicle state
        opti.subject_to(t[0] == self.nearest_wp_idx)                     

        opti.subject_to(opti.bounded(-1.0, u[0, :], 1.0))       # Bounded steering
        opti.subject_to(opti.bounded(-1.22, u[1, :], 1.22))     # Bounded throttle/brake
        opti.subject_to(opti.bounded(0, t, max_L))              # Bounded path length
        opti.subject_to(opti.bounded(0, v, self.vmax))          # Bounded path progression
        
        # Prediction till control horizon
        for i in range(self.C):
            if i < 0.6 * self.P:
                dt = Ts
            else:
                dt = Ts + 0

            opti.subject_to(s[:, i+1] == pred(s[:, i], u[:, i], dt))
            opti.subject_to(e[:, i] == self.calculate_error(s[:, i], t[i]))
            opti.subject_to(t[i + 1] == t[i] + v[i] * dt)

        # Prediction after control horizon (constant control from last step)
        for i in range(self.C, self.P):
            if i < 0.6 * self.P:
                dt = Ts
            else:
                dt = Ts + 0

            opti.subject_to(s[:, i+1] == pred(s[:, i], u[:, self.C - 1], dt))
            opti.subject_to(e[:, i] == self.calculate_error(s[:, i], t[i]))
            opti.subject_to(t[i + 1] == t[i] + v[i] * dt)

        opti.subject_to(e[:, -1] == self.calculate_error(s[:, -1], t[-1]))

        # Variable Initializations
        predicted_last_state = self.predict_new(prev_states[:, -1], prev_controls[:, -1], system_params, dt)
        opti.set_initial(s, np.vstack((self.car_state, prev_states[:, 2:].T, predicted_last_state)).T)
        opti.set_initial(u, np.vstack((prev_controls[:, 1:].T, prev_controls[:, -1])).T)
        opti.set_initial(t, np.hstack((prev_t[1:], prev_t[-1] + prev_v[-1] * dt)))
        opti.set_initial(v, np.hstack((prev_v[1:], prev_v[-1])))
        opti.set_initial(e, 0)

        # opti.set_initial(s, np.vstack([self.car_state] * (self.P + 1)).T)
        # opti.set_initial(u, np.vstack([prev_controls[:, 0]] * self.C).T)
        # opti.set_initial(t, self.nearest_wp_idx + np.arange(0, self.P + 1))
        # opti.set_initial(v, self.car_state[3])

        p_opts = {"print_time": False, 'ipopt.print_level': 0, "ipopt.expect_infeasible_problem": "yes", "ipopt.max_iter": 100}
        opti.solver('ipopt', p_opts)

        sol = opti.solve()


        opti_states = sol.value(s)
        opti_controls = sol.value(u)
        opti_errors = sol.value(e)
        opti_t = sol.value(t)
        opti_v = sol.value(v)

        # Plot predicted states
        # plt.plot(opti_states[0, :], opti_states[1, :])
        # plt.pause(0.1)

        # print('Predicted states: \n {} \n'.format(opti_states))
        # print('Contouring and Lag errors: \n {} \n'.format(opti_errors))
        # print('Optimized Controls: \n {} \n'.format(opti_controls))
        # print('Predicted parameter: \n {} \n'.format(opti_t))
        # print('Path progression: \n {} \n'.format(opti_v))

        return opti_states, opti_controls, opti_t, opti_v

def main():
    try:
        # Data logging flag
        save_flag = False

        # Initialize car environment
        env = CarEnv()

        # Load waypoints
        waypoints = env.read_file("../../Data/2D_waypoints.txt")
        system_params = env.read_file('../../Data/params.txt')

        # Spawn a vehicle at spawn_pose
        spawn_idx = 0
        env.nearest_wp_idx = spawn_idx
        env.car_state = env.spawn_vehicle_2D(spawn_idx, waypoints)
        for i in range(50):
            env.world.tick()

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

        # Append initial state and controls
        # curr_t = env.world.wait_for_tick().timestamp.elapsed_seconds # [seconds]
        
        env.controls.append(control(0, 0.0, 0.0, 0.0, 0.0, 0))

        env.lut_x, env.lut_y, env.lut_theta = env.fit_curve(waypoints)

        # Set controller tuning params
        env.set_mpc_params(P = 25, C = 25, vmax = 25)
        env.set_opti_weights(w_u0 = 1, w_u1 = 1, w_qc = 0.75, w_ql = 2, gamma = 10, w_c = 2)
        Ts = 0.1

        prev_states = np.vstack([env.car_state] * (env.P + 1)).T
        prev_controls = np.zeros((2, env.C))
        prev_t = np.ones(env.P + 1) * env.nearest_wp_idx
        prev_v = np.zeros(env.P + 1)

        # Initialize control loop
        while(1):
            env.world.tick()
            env.car_state = env.get_true_state()
            print(env.car_state[3])
            env.nearest_wp_idx = env.calculate_nearest_index(env.car_state, waypoints)
            try:
                prev_states, prev_controls, prev_t, prev_v = env.mpc(prev_states, prev_controls, prev_t, prev_v, system_params, waypoints.shape[0] + 1, Ts)
            except RuntimeError as err:
                print('Error occured with following error message: \n {} \n'.format(err))
                pass
            if prev_controls[0, 0] > 0:
                throttle = prev_controls[0, 0]
                brake = 0
            else:
                brake = prev_controls[0, 0]
                throttle = 0

            steer = prev_controls[1, 0] / 1.22

            # env.states.append(state(0, env.car_state[0], env.car_state[1], env.car_state[2], env.car_state[3], env.car_state[4], 0))
            # env.controls.append(control(0, throttle, brake, env.control[0], steer, 0))

            env.vehicle.apply_control(carla.VehicleControl(
                throttle = throttle, steer = steer, reverse = False, brake = brake))

            if (env.nearest_wp_idx == waypoints.shape[0] - 1) and env.nearest_wp_idx != prev_idx:
                laps_completed += 1
                if laps_completed == num_of_laps:
                    while env.car_state[3] != 0.0:
                        env.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0, reverse = False, brake = 0.2))
                        env.car_state = env.get_true_state()
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