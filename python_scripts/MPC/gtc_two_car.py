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

PI_2 = np.pi * 2

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
        self.states_1 = []
        self.states_2 = []

        # Initialize control dataclass list to log subsequent control commands
        self.controls_1 = []
        self.controls_2 = []

        # Nearest Waypoints Index
        self.nearest_idx_c1 = 0
        self.nearest_idx_c2 = 0

        # Vehicle Current Pose
        self.car_1_state = np.empty(5)
        self.car_2_state = np.empty(5)

        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []

        # Control Command
        self.control_1 = np.array([0, 0])
        self.control_2 = np.array([0, 0])

    def spawn_vehicle_2D(self, spawn_idx: int, waypoints: np.ndarray, offset: float) -> (np.ndarray):
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

    def get_true_state(self, vehicle, orient_flag: bool) -> (np.ndarray):
        """Get vehicles true state from the simulation
        
        Returns
        -------
        - true_state: True state of the vehicle [x, y, yaw, vx, vy]
        """
        x = vehicle.get_transform().location.x
        y = vehicle.get_transform().location.y

        if orient_flag:
            yaw = ((np.radians(vehicle.get_transform().rotation.yaw)) + PI_2) % PI_2    # [radians]
        else:
            yaw = wrapToPi(np.radians(vehicle.get_transform().rotation.yaw))    # [radians]

        vx = vehicle.get_velocity().x * np.cos(yaw) + vehicle.get_velocity().y * np.sin(yaw)
        vy = -vehicle.get_velocity().x * np.sin(yaw) + vehicle.get_velocity().y * np.cos(yaw)

        true_state = np.array([x, y, yaw, vx, vy])

        return true_state

    def predict_new(self, orient_flag, old_state: np.ndarray, control: np.ndarray, params: np.ndarray, dt: float) -> (np.ndarray):
        L, p, Cd, Cf, Cc = params

        x, y, theta, vx, vy = old_state
        v = np.sqrt(vx ** 2 + vy ** 2)

        acc, delta = control

        x_new = x + (v * np.cos(np.arctan2(np.tan(delta), 2) + theta) * dt)
        y_new = y + (v * np.sin(np.arctan2(np.tan(delta), 2) + theta) * dt)
        if orient_flag:
            theta_new = ((theta + (v * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2)))) + PI_2) % PI_2
        else:
            theta_new = wrapToPi(theta + (v * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
        vx_new = vx + (p * acc - Cd * v * vx - Cf * vx) * dt
        vy_new = vy - (Cc * (np.arctan2(vy, vx) - delta) + (Cd * v + Cf) * vy) * dt

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
        # Extract vehicle 1 position
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

        e_l = -cos(yaw) * (x - x_ref) - sin(yaw) * (y - y_ref)

        return e_l

    def set_mpc_params(self, P: int, vmax: float) -> NoReturn:
        self.P = P
        self.vmax = vmax
        
    def set_opti_weights(self, weights: dict) -> NoReturn:
        self.w_u0 = weights['w_u0']
        self.w_u1 = weights['w_u1']
        self.w_lag = weights['w_lag']
        self.gamma = weights['gamma']
        self.w_c = weights['w_c']
        self.w_ds = weights['w_ds']

    def w_matrix(self,init_value,step_size):
        w = np.eye(self.P + 1)
        for i in range(self.P + 1):
            w[i, i] = init_value + step_size * i
        return w

    def fit_curve(self, waypoints: np.ndarray, L: np.ndarray) -> Tuple[Function, Function, Function, Function]:
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
        # Waypoints interpolation
        self.lut_x = interpolant('LUT_x', 'bspline', [L], waypoints[:, 0], dict(degree=[3]))
        self.lut_y = interpolant('LUT_y', 'bspline', [L], waypoints[:, 1], dict(degree=[3]))
        self.lut_theta = interpolant('LUT_t', 'bspline', [L], waypoints[:, 2], dict(degree=[3]))

        # Soft constraint cost for track boundaries
        t = 6                               # Threshold
        cost_fit = np.zeros((10000))
        numbers = np.linspace(-16,16,10000)
        for i in range(10000):
            if -t <= numbers[i] <= t:
                cost_fit[i] = 0.0
            else:
                cost_fit[i] = (abs(numbers[i]) - t) ** 2
        self.lut_d = interpolant('LUT_d', 'bspline', [numbers], cost_fit, dict(degree=[3]))

        return self.lut_x, self.lut_y, self.lut_theta, self.lut_d

    def track_constraints(self, state_mpc: MX, theta: MX) -> (MX): 
        x = state_mpc[0]
        y = state_mpc[1]

        x_ref = self.lut_x(theta)
        y_ref = self.lut_y(theta)
        yaw = self.lut_theta(theta)
        
        track_width = 15.0           # [m]
        d = (track_width * 0.8)/2
        
        a = -tan(yaw)
        b = 1
        c = (tan(yaw) * x_ref) - y_ref

        c1 = c - (d * sqrt(1 + (tan(yaw) ** 2)))
        c2 = c + (d * sqrt(1 + (tan(yaw) ** 2)))

        u_b = (a * x  + b * y + c1) / sqrt(a ** 2 + b ** 2)
        l_b = (a * x  + b * y + c2) / sqrt(a ** 2 + b ** 2) 

        cost = self.lut_d(u_b + l_b)

        bounds = MX(vcat([cost, u_b + l_b]))
        
        return bounds

    def dist_to_ellipse(self, ref_states: np.ndarray, state_mpc: MX, a: float, b: float) -> (MX):
        x_c = ref_states[0]
        y_c = ref_states[1]
        theta = -ref_states[2]

        x = state_mpc[0]
        y = state_mpc[1]

        d = ((((x_c - x) * cos(theta) - (y_c - y) * sin(theta)) ** 2) / (a ** 2)) + ((((x_c - x) * sin(theta) + (y_c - y) * cos(theta)) ** 2) / (b ** 2))

        return d

    def stanley_control(self, waypoints: np.ndarray, car_state: np.ndarray, max_steer_angle: float) -> (float):
        x_des, y_des, yaw_des = waypoints[self.nearest_idx_c1 + 3]
        x, y, yaw, v_lon, v_lat = car_state

        d = np.sqrt((x - x_des) ** 2 + (y - y_des) ** 2)

        # Heading error [radians]
        psi_h = wrapToPi(yaw_des - yaw)

        # Crosstrack yaw difference to path yaw [radians]
        yaw_diff = wrapToPi(yaw_des - np.arctan2(y - y_des, x - x_des))

        # Crosstrack error in yaw [radians]
        psi_c = np.arctan2(0.5 * np.sign(yaw_diff) * d, 5.0 + v_lon)

        # Steering angle control
        steer = np.degrees(wrapToPi(psi_h + psi_c))                 # uncontrained in degrees
        steer = max(min(steer, max_steer_angle), -max_steer_angle)
        steer = (steer) / max_steer_angle                           # constrained to [-1, 1]

        return steer

    def mpc(self, orient_flag: bool, prev_vals: dict, curr_state: np.ndarray, nearest_idx: int, obs_vals: dict, system_params: np.ndarray, max_L: float, L: np.ndarray, Ts: float) -> (dict):

        """Model Predictive Controller Function
        Arguments
        ---------
        - prev_states: vehicle state predictions from the previous iteration [5 x P+1]
        - prev_controls: controls computed in the previous iteration [2 x P]
        - prev_t: path length parameter from the previous iteration [1 x P+1]
        - prev_v: path progression parameter from the previous iteration [1 x P+1]
        - system_params: Vehicle dynamics parameters obtained from System ID
        - max_L: maximum path length of the track
        - Ts: Sampling Time of the controller
        Returns
        -------
        - opti_states: vehicle state predictions from the current iteration [5 x P+1]
        - opti_controls: optimal controls computed in this iteration [2 x P]
        - opti_t: path length parameter from this iteration [1 x P+1]
        - opti_v: path progression parameter from the current iteration [1 x P+1]
        """        
        
        ##### Define state dynamics in terms of casadi function #####
        l, p, Cd, Cf, Cc = system_params
        d_nan = 1e-5 # Add small number to avoid NaN error at zero velocities in the Jacobian evaluation

        dt = MX.sym('dt')
        state = MX.sym('s', 5)
        control_command = MX.sym('u', 2)

        x, y, yaw, vx, vy = state[0], state[1], state[2], state[3], state[4]
        acc, delta = [control_command[0], control_command[1]]

        if orient_flag:
            state_prediction = vertcat(x + sqrt((vx + d_nan) ** 2 + (vy + d_nan)**2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                                y + sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                                fmod(((yaw + (sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * tan(delta) * dt / sqrt((l ** 2) + (0.5 * l * tan(delta))**2))) + PI_2), PI_2),
                                vx + ((p * acc) - (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * vx - Cf * vx)) * dt,
                                vy - (Cc * (atan2(vy, vx + d_nan) - delta) + (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) + Cf) * vy) * dt)
        else:
            state_prediction = vertcat(x + sqrt((vx + d_nan) ** 2 + (vy + d_nan)** 2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                                y + sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                                (yaw + (sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * tan(delta) * dt / sqrt((l ** 2) + (0.5 * l * tan(delta)) ** 2))),
                                vx + ((p * acc) - (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * vx - Cf * vx)) * dt,
                                vy - (Cc * (atan2(vy, vx + d_nan) - delta) + (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) + Cf) * vy) * dt)

        pred = Function('pred', [state, control_command, dt], [state_prediction])

        prev_states = prev_vals['states']
        prev_controls = prev_vals['controls']
        prev_t = prev_vals['t']
        prev_v = prev_vals['v']

        ##### Optistack begin #####
        opti = Opti()

        mpc_states = opti.variable(5, self.P + 1)       # States [x, y, yaw, vx, vy]
        lag_error = opti.variable(1, self.P + 1)        # Lag Error
        t = opti.variable(1, self.P + 1)                # Path length parameter
        v = opti.variable(1, self.P + 1)                # Path progression rate
        u = opti.variable(2, self.P)                    # Controls [throttle/brake, steering]
        d_s = opti.variable(1, self.P)                  # Cost for approaching track boundaries
        c = opti.variable(1, self.P)

        # Costs to optimize over
        p_acc   =  u[0, :]  @ self.w_u0 @ u[0, :].T                             # Throttle/Brake cost
        p_steer =  u[1, :] @  self.w_u1 @ u[1, :].T                             # Steering cost
        p_control_roc = self.w_c * sumsqr(u[:, :self.P-1] - u[:, 1:self.P])     # Rate of Change of Controls

        p_v_roc = self.w_c * sumsqr(v[:self.P] - v[1:self.P + 1])               # Rate of Change of progression rate 
        r_v_max = ((v * Ts) @ self.gamma @ (v * Ts).T)                          # Progression Reward
        p_lag = lag_error @ self.w_lag @ lag_error.T                            # Lag error
        c_d_s  = d_s @ self.w_ds @ d_s.T                                        # Soft distance cost

        # Minimization objective
        opti.minimize(p_lag + p_control_roc + p_v_roc + p_acc + p_steer - r_v_max + 1.5 * sumsqr(c[1:] - c[:-1]) + c_d_s)
        
        # Constraints

        opti.subject_to(mpc_states[:, 0] == curr_state)              # Start prediction with true vehicle state
        opti.subject_to(t[0] == L[nearest_idx])                     

        opti.subject_to(opti.bounded(-1.0, u[0, :], 1.0))       # Bounded steering
        opti.subject_to(opti.bounded(-1.22, u[1, :], 1.22))     # Bounded throttle/brake
        opti.subject_to(opti.bounded(0, t, max_L + 50))         # Bounded path length
        opti.subject_to(opti.bounded(0, v, self.vmax))          # Bounded path progression
        
        # Prediction horizon
        for i in range(self.P):
            if i < 0.6 * self.P:
                dt = Ts
            else:
                dt = Ts + 0.1

            opti.subject_to(mpc_states[:, i+1] == pred(mpc_states[:, i], u[:, i], dt))
            opti.subject_to(c[i] == self.track_constraints(mpc_states[:, i + 1], t[i + 1])[1])
            opti.subject_to(d_s[i] == self.track_constraints(mpc_states[:, i + 1], t[i + 1])[0])
            opti.subject_to(lag_error[i] == self.calculate_error(mpc_states[:, i], t[i]))
            opti.subject_to(t[i + 1] == t[i] + v[i] * dt)

            d = self.dist_to_ellipse(obs_vals['states'][:, i + 1], mpc_states[:, i + 1], 4.5, 3)
            opti.subject_to(d > 1)

        opti.subject_to(lag_error[-1] == self.calculate_error(mpc_states[:, -1], t[-1]))

        # Variable Initializations
        if orient_flag:
            opti.set_initial(mpc_states, np.vstack([curr_state] * (self.P + 1)).T)
        else:
            predicted_last_state = self.predict_new(orient_flag, prev_states[:, -1], prev_controls[:, -1], system_params, dt)
            opti.set_initial(mpc_states, np.vstack((curr_state, prev_states[:, 2:].T, predicted_last_state)).T)
        opti.set_initial(u, np.vstack((prev_controls[:, 1:].T, prev_controls[:, -1])).T)
        opti.set_initial(t, np.hstack((prev_t[1:], prev_t[-1] + prev_v[-1] * dt)))
        opti.set_initial(v, np.hstack((prev_v[1:], prev_v[-1])))
        opti.set_initial(d_s, 0)
        opti.set_initial(lag_error, 0)

        # Set ipopt options
        p_opts = {"print_time": False, 'ipopt.print_level': 0, "ipopt.expect_infeasible_problem": "yes", "ipopt.max_iter": 175}
        opti.solver('ipopt', p_opts)

        sol = opti.solve()

        opti_states = sol.value(mpc_states)
        opti_controls = sol.value(u)
        opti_errors = sol.value(lag_error)
        opti_t = sol.value(t)
        opti_v = sol.value(v)

        prev_vals['states'] = opti_states
        prev_vals['controls'] = opti_controls
        prev_vals['t'] = opti_t
        prev_vals['v'] = opti_v

        plot_flag = False
        if plot_flag:
            # Plot predicted states
            plt.plot(opti_states[0, :], opti_states[1, :])
            plt.pause(0.01)
            plt.cla()

        print_flag = False
        if print_flag:
            print('Predicted states: \n {} \n'.format(opti_states[2, :]))
            print('Optimized Controls: \n {} \n'.format(opti_controls))
            print('Predicted parametewer: \n {} \n'.format(opti_t))
            print('Path progression: \n {} \n'.format(opti_v))

        return prev_vals

def main():
    try:
        # Data logging flag
        save_flag = True

        # Initialize car environment
        env = CarEnv()

        # Load waypoints
        waypoints = env.read_file("../../Data/2D_waypoints.txt")
        sys_params = env.read_file('../../Data/params.txt')

        # Spawn a vehicle at spawn_pose
        spawn_idx = 200
        env.nearest_idx_c1 = spawn_idx
        env.nearest_idx_c2 = spawn_idx
        vehicle_1, env.car_1_state, max_steer_angle_1 = env.spawn_vehicle_2D(spawn_idx + 5, waypoints, -4)
        vehicle_2, env.car_2_state, max_steer_angle_2 = env.spawn_vehicle_2D(spawn_idx, waypoints, 4)

        env.states_1.append(state(0, env.car_1_state[0], env.car_1_state[1], env.car_1_state[2], env.car_1_state[3], env.car_1_state[4], 0))
        env.states_2.append(state(0, env.car_2_state[0], env.car_2_state[1], env.car_2_state[2], env.car_2_state[3], env.car_2_state[4], 0))

        for i in range(50):
            env.world.tick()

        # Default params
        end_flag = [False, False]
        num_of_laps = [1, 1]
        prev_idx = [0, 0]
        laps_completed = [0, 0]

        # Read command line args
        argv = sys.argv[1:]
        if len(argv) != 0:
            opts, args = getopt.getopt(argv, shortopts='n:s:', longopts=['nlaps', 'save'])
            for tup in opts:
                if tup[0] == '-n':
                    num_of_laps[:] = int(tup[1])
                    save_flag = True

        # Append initial state and controls
        # curr_t = env.world.wait_for_tick().timestamp.elapsed_seconds # [seconds]
        
        env.controls_1.append(control(0, 0.0, 0.0, 0.0, 0.0, 0))
        env.controls_2.append(control(0, 0.0, 0.0, 0.0, 0.0, 0))

        waypoints_ext = np.vstack((waypoints, waypoints[:50, :]))
        l = np.cumsum(np.sqrt(np.sum(np.square(waypoints_ext[:-1, :2] - waypoints_ext[1:, :2]), axis = 1)))
        L = np.r_[0, l]
        max_L = L[waypoints.shape[0] + 1]
        env.lut_x, env.lut_y, env.lut_theta, env.lut_d = env.fit_curve(waypoints_ext, L)

        # Set controller tuning params
        env.set_mpc_params(P = 30, vmax = 35)  
        weights = {}
        weights['w_u0'] = env.w_matrix(1, 0)[:-1, :-1]
        weights['w_u1'] = env.w_matrix(2, 0)[:-1, :-1]
        weights['w_lag'] = env.w_matrix(3, -0.02)
        weights['gamma'] = env.w_matrix(8, -0.2)
        weights['w_ds'] = env.w_matrix(init_value=1, step_size=1)[:-1, :-1]
        weights['w_c'] = 5
        env.set_opti_weights(weights)     
        Ts = 0.1

        prev_states_1 = np.vstack([env.car_1_state] * (env.P + 1)).T
        prev_controls_1 = np.zeros((2, env.P))
        prev_controls_1[0, :] = prev_controls_1[0, :] + 0.5
        prev_t_1 = np.ones(env.P + 1) * env.nearest_idx_c1
        prev_v_1 = np.zeros(env.P + 1)

        prev_vals_1 = {}
        prev_vals_1['states'] = prev_states_1
        prev_vals_1['controls'] = prev_controls_1
        prev_vals_1['t'] = prev_t_1
        prev_vals_1['v'] = prev_v_1

        prev_states_2 = np.vstack([env.car_2_state] * (env.P + 1)).T
        prev_controls_2 = np.zeros((2, env.P))
        prev_controls_2[0, :] = prev_controls_2[0, :] + 0.5
        prev_t_2 = np.ones(env.P + 1) * env.nearest_idx_c2
        prev_v_2 = np.zeros(env.P + 1)

        prev_vals_2 = {}
        prev_vals_2['states'] = prev_states_2
        prev_vals_2['controls'] = prev_controls_2
        prev_vals_2['t'] = prev_t_2
        prev_vals_2['v'] = prev_v_2

        total_iterations = 0
        fallbacks_1 = 0
        fallbacks_2 = 0
        # Initialize control loop
        iters = 3
        while(1):
            total_iterations += 1
            if ((total_iterations % 50) == 0):
                        print('Success rate of MPC car 1 = ', (total_iterations - fallbacks_1) / total_iterations)
                        print('Success rate of MPC car 2 = ', (total_iterations - fallbacks_2) / total_iterations)

            if (1500 > env.nearest_idx_c1 > 1250):
                orient_flag_1 = True
            else:
                orient_flag_1 = False
            
            if (1500 > env.nearest_idx_c2 > 1250):
                orient_flag_2 = True
            else:
                orient_flag_2 = False

            env.car_1_state = env.get_true_state(vehicle_1, orient_flag_1)
            env.nearest_idx_c1 = env.calculate_nearest_index(env.car_1_state, waypoints)
            env.car_2_state = env.get_true_state(vehicle_2, orient_flag_2)
            env.nearest_idx_c2 = env.calculate_nearest_index(env.car_2_state, waypoints)
            print(env.car_1_state[3], env.car_2_state[3])

            # env.world.debug.draw_box(carla.BoundingBox(carla.Location(env.car_1_state[0], env.car_1_state[1], 0), carla.Vector3D(4.5, 3, 1)), carla.Rotation(0, np.degrees(env.car_1_state[2]), 0), thickness=0.1, color=carla.Color(0, 255, 0), life_time=1)
            # env.world.debug.draw_box(carla.BoundingBox(carla.Location(env.car_2_state[0], env.car_2_state[1], 0), carla.Vector3D(4.5, 3, 1)), carla.Rotation(0, np.degrees(env.car_2_state[2]), 0), thickness=0.1, color=carla.Color(0, 255, 0), life_time=1)

            

            # env.cam_sensor.listen(lambda image: image.save_to_disk('/home/dhagash/MS-GE-02/MSR-Project/camera_pos_fix/%06d.png' % image.frame))
            
            for i in range(iters):
               
                if not end_flag[0]:
                    try:
                        
                        
                        # env.set_mpc_params(P = 30, vmax = 25)

                        prev_vals_1 = env.mpc(orient_flag_1, prev_vals_1, env.car_1_state, env.nearest_idx_c1, prev_vals_2, sys_params, max_L, L, Ts)
                        
                        steer_1 = prev_vals_1['controls'][1, 0] / 1.22

                        

                    except RuntimeError as err:
                        
                        if i < iters-1:
                            pass
                        
                        else:

                            fallbacks_1 += 1
                            
                            print('Car 1: Error occured with following error message: \n {} \n'.format(err))
                            print('Car 1: Fallback on Stanley Control !!!')
                            steer_1 = env.stanley_control(waypoints, env.car_1_state, max_steer_angle_1)
                            prev_vals_1['controls'][0, :] = prev_vals_1['controls'][0, 0]
                            prev_vals_1['controls'][1, :] = steer_1
                            prev_vals_1['states'] = np.vstack([env.car_1_state] * (env.P + 1)).T
                            prev_vals_1['t'] = prev_vals_1['t'] + env.car_1_state[3] * Ts
                            pass

                    if prev_controls_1[0, 0] > 0:
                        throttle_1 = prev_vals_1['controls'][0, 0]
                        brake_1 = 0
                    else:
                        brake_1 = prev_vals_1['controls'][0, 0]
                        throttle_1 = 0              

                    

                            
            
                
                if not end_flag[1]:
                    try:

                        # env.set_mpc_params(P = 30, vmax = 35)
                        prev_vals_2 = env.mpc(orient_flag_2, prev_vals_2, env.car_2_state, env.nearest_idx_c2, prev_vals_1, sys_params, max_L, L, Ts)
                        steer_2 = prev_vals_2['controls'][1, 0] / 1.22

                    except RuntimeError as err:
                        if i < iters-1:
                            pass
                        
                        else:
                            fallbacks_2 += 1
                            print('Car 2: Error occured with following error message: \n {} \n'.format(err))
                            print('Car 2: Fallback on Stanley Control !!!')
                            steer_2 = env.stanley_control(waypoints, env.car_2_state, max_steer_angle_2)
                            prev_vals_2['controls'][0, :] = prev_vals_2['controls'][0, 0]
                            prev_vals_2['controls'][1, :] = steer_2
                            prev_vals_2['states'] = np.vstack([env.car_2_state] * (env.P + 1)).T
                            prev_vals_2['t'] = prev_vals_2['t'] + env.car_2_state[3] * Ts
                            pass

                    if prev_vals_2['controls'][0, 0] > 0:
                        throttle_2 = prev_vals_2['controls'][0, 0]
                        brake_2 = 0
                    else:
                        brake_2 = prev_vals_2['controls'][0, 0]
                        throttle_2 = 0
                
                
              
            #     plt.plot(prev_vals_1['states'][0,:],prev_vals_1['states'][1,:],'r')
            #     plt.plot(prev_vals_2['states'][0,:],prev_vals_2['states'][1,:],'b')
            #     plt.pause(1)
            
            # plt.cla()
                # plt.show()

                
            if (env.nearest_idx_c1 == waypoints.shape[0] - 1) and env.nearest_idx_c1 != prev_idx[0]:
                laps_completed[0] += 1
                prev_vals_1['t'] = prev_vals_1['t'] - prev_vals_1['t'][0]
                if laps_completed[0] == num_of_laps[0]:
                    end_flag[0] = True
                    throttle_1 = 0
                    steer_1 = 0
                    brake_1 = 0.2

            prev_idx[0] = env.nearest_idx_c1

            if (env.nearest_idx_c2 == waypoints.shape[0] - 1) and env.nearest_idx_c2 != prev_idx[1]:
                laps_completed[1] += 1
                prev_vals_2['t'] = prev_vals_2['t'] - prev_vals_2['t'][0]
                if laps_completed[1] == num_of_laps[1]:
                    end_flag[1] = True
                    throttle_2 = 0
                    steer_2 = 0
                    brake_2 = 0.2

            prev_idx[1] = env.nearest_idx_c2

            # print("Applied Control xD")

            vehicle_1.apply_control(carla.VehicleControl(throttle = throttle_1, steer = steer_1, reverse = False, brake = brake_1))
            vehicle_2.apply_control(carla.VehicleControl(throttle = throttle_2, steer = steer_2, reverse = False, brake = brake_2))
            env.world.tick()

            env.states_1.append(state(0, env.car_1_state[0], env.car_1_state[1], env.car_1_state[2], env.car_1_state[3], env.car_1_state[4], 0))
            env.controls_1.append(control(0, throttle_1, brake_1, env.control_1[0], steer_1, 0))

            env.states_2.append(state(0, env.car_2_state[0], env.car_2_state[1], env.car_2_state[2], env.car_2_state[3], env.car_2_state[4], 0))
            env.controls_2.append(control(0, throttle_2, brake_2, env.control_2[0], steer_2, 0))

            if env.car_1_state[3] == 0.0 and env.car_2_state[3] == 0 and laps_completed[0] == num_of_laps[0] and laps_completed[1] == num_of_laps[1]:
                break

    finally:

        if save_flag:

        # Save all Dataclass object lists to a file
            
            env.save_log('../../Data/states_1_gtc_test.pickle', env.states_1)
            env.save_log('../../Data/controls_1_gtc_test.pickle', env.controls_1)
            env.save_log('../../Data/states_2_gtc_test.pickle', env.states_2)
            env.save_log('../../Data/controls_2_gtc_test.pickle', env.controls_2)

        # Destroy all actors in the simulation
        env.destroy()

if __name__=="__main__":
    main()
