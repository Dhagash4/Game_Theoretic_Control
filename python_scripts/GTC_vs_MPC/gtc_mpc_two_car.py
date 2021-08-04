#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project: Game Theoretic Control for Multi Robot Racing

# Game Theoretic Control v/s Model Predictive Control for Two Cars

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import glob
import os, sys
import argparse
from typing import Tuple, NoReturn

import casadi
from casadi import *

import numpy as np
from numpy.lib.utils import info
import _pickle as cpickle
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.util import *

#Import CARLA anywhere on the system
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
# ==============================================================================

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
        self.settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(self.settings)

        # Get Map of the world
        self.map = self.world.get_map()
        
        # Get all the blueprints available in the world
        self.blueprint_library =  self.world.get_blueprint_library()

        # Nearest Waypoints Index
        self.nearest_idx_c1 = 0
        self.nearest_idx_c2 = 0

        # Vehicle Current Pose
        self.car_1_state = np.empty(5)
        self.car_2_state = np.empty(5)

        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []

    def spawn_vehicle_2D(self, spawn_idx: int, waypoints: np.ndarray, offset: float) -> Tuple[carla.libcarla.Vehicle, np.ndarray, float]:
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

    def CameraSensor(self) -> NoReturn:
        """Attach a camera to the vehicle for recording images from car POV
        """
        # RGB Camera Sensor 
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x','640')
        self.rgb_cam.set_attribute('image_size_y', '480')
        self.rgb_cam.set_attribute('fov', '110')
        
        # Attaching sensor to car
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

    def get_true_state(self, vehicle: carla.libcarla.Vehicle, orient_flag: bool) -> (np.ndarray):
        """Get vehicle's true state from the simulation
        
        Arguments
        ---------
        - vehicle: carla instance of the desired vehicle
        - orient_flag: indicates the region where orientation jumps between[-pi] and [+pi] occur

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

    def predict_new(self, orient_flag: bool, old_state: np.ndarray, control: np.ndarray, sys_params: np.ndarray, dt: float) -> (np.ndarray):
        """Generates a prediction of the new vehicle states given the old state and the controls applied
        
        Arguments
        ---------
        - orient_flag: indicates the region where orientation jumps between[-pi] and [+pi] occur
        - old_state: old state of the vehicle [x, y, yaw, vx, vy]
        - control: control commands given to the vehicle [throttle/brake, steer]
        - sys_params: vehicle dynamics parameters obtained from System ID
        - dt: sampling time of the controller

        Returns
        -------
        - new_state: new predicted state of the vehicle [x, y, yaw, vx, vy]
        """
        L, p, Cd, Cf, Cc = sys_params

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

    def const_vel_model(self, start_state: np.ndarray, dt: float, N: int) -> (np.ndarray):
        predictions = np.zeros((5, N + 1))
        predictions[:, 0] = start_state

        for i in range(1, N + 1):
            x, y, theta, vx, vy = predictions[:, i - 1]
            v = np.sqrt(vx ** 2 + vy ** 2)
        
            x_new = x + (v * np.cos(theta) * dt)
            y_new = y + (v * np.sin(theta) * dt)
            
            new_state = np.array([x_new, y_new, theta, vx, vy])

            predictions[:, i] = new_state

        return predictions

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
        dx = (waypoints[:, 0] - x_r) ** 2
        dy = (waypoints[:, 1] - y_r) ** 2
        dist = dx + dy

        # Find index corresponding to minimum distance
        idx = np.argmin(dist)

        return idx

    def calculate_error(self, state_mpc: casadi.MX, theta: casadi.MX) -> (casadi.MX):
        """Compute Contouring and Lag errors at given prediction step for the Model Predictive Contour Controller

        Arguments
        ---------
        - state_mpc: predicted state by the MPC loop (MX array [x, y, yaw, vx, vy])
        - theta: predicted path length parameter

        Returns
        -------
        - e_lag: Lag error
        """
        x = state_mpc[0]
        y = state_mpc[1]

        x_ref   = self.lut_x(theta)
        y_ref   = self.lut_y(theta)
        yaw     = self.lut_theta(theta)

        e_lag = -cos(yaw) * (x - x_ref) - sin(yaw) * (y - y_ref)

        return e_lag

    def set_mpc_params(self, P: int = 25, vmax: float = 35) -> NoReturn:
        """Set parameters needed for the Model Predictive controller
        
        Arguments
        ---------
        - P: Prediction horizon of the controller
        - vmax: maximum allowed velocity for the vehicle
        """
        self.P = P
        self.vmax = vmax
        
    def w_matrix(self, init_value: float, step_size: float) -> (np.ndarray):
        """Setup the weight matrix with given initial values and step size

        Arguments
        ---------
        - init_value: weight value for the starting iteration of MPC
        - step_size: step size for subsequent weights

        Returns
        -------
        - w: diagonal weight matrix to be used in the cost function
        """
        w = np.array([init_value + step_size * i for i in range(self.P + 1)])
        
        return np.diag(w)

    def fit_curve(self, waypoints: np.ndarray, L: np.ndarray) -> Tuple[casadi.Function, casadi.Function, casadi.Function, casadi.Function, casadi.Function]:
        """Fit 3 degree b-splines to be used by casadi optistack

        Arguments
        ---------
        - waypoints: pre-computed waypoints to fit a parametric spline to
        - L: cumulative sum of distance between 2 consecutive waypoints to be used as a parameter for the splines

        Returns
        -------
        - lut_x: Look up table for the x coordinate
        - lut_y: Look up table for the y coordinate
        - lut_theta: Look up table for the path orientation
        - lut_boundary_cost: Look up table for the track boundary constraint costs
        - lut_collision_cost: Look up table for the collision constraint costs
        """
        # Waypoints interpolation
        self.lut_x = interpolant('LUT_x', 'bspline', [L], waypoints[:, 0], dict(degree=[3]))
        self.lut_y = interpolant('LUT_y', 'bspline', [L], waypoints[:, 1], dict(degree=[3]))
        self.lut_theta = interpolant('LUT_t', 'bspline', [L], waypoints[:, 2], dict(degree=[3]))

        # Soft constraint cost for track boundaries
        t = 6                           # Threshold
        cost_fit = np.zeros((10000))
        numbers = np.linspace(-16,16,10000)
        for i in range(10000):
            if -t <= numbers[i] <= t:
                cost_fit[i] = 0.0
            else:
                cost_fit[i] = (abs(numbers[i]) - t) ** 2
        self.lut_boundary_cost = interpolant('LUT_boundary_cost', 'bspline', [numbers], cost_fit, dict(degree=[3]))

        # Soft constraint cost for collision
        cost_fit = np.zeros((100000))
        numbers = np.linspace(0, 2500, 100000)
        for i in range(100000):
            if numbers[i] >= 1:
                cost_fit[i] = 0.0
            else:
                cost_fit[i] = ((1 - numbers[i]) * 1) ** 3
        self.lut_collision_cost = interpolant('LUT_collision_cost', 'bspline', [numbers], cost_fit, dict(degree=[3]))

        return self.lut_x, self.lut_y, self.lut_theta, self.lut_boundary_cost, self.lut_collision_cost

    def track_constraints(self, state_mpc: casadi.MX, theta: casadi.MX) -> (casadi.MX): 
        """Compute the track constraint violation cost and the vehicle position relative to the track centerline
        
        Arguments
        ---------
        - state_mpc: state predicted by the mpc at any given prediction horizon
        - theta: path length predicted by the MPC at the same prediction step

        Returns
        -------
        A casadi array containing the cost for constraint violation and the relative position of the car wrt centerline
        """
        # Current position
        x = state_mpc[0]
        y = state_mpc[1]

        # Reference pose of the nearest position on the track
        x_ref   = self.lut_x(theta)
        y_ref   = self.lut_y(theta)
        yaw     = self.lut_theta(theta)
        
        track_width = 15.0              # Width of the given track in meters
        d = (track_width * 0.8)/2       # Keep buffer of 20%
        
        # Straight line equation parameters passing through the reference position at computed orientation
        a = -tan(yaw)
        b = 1
        c = (tan(yaw) * x_ref) - y_ref

        # Constant parameters for two parallel lines at offset +d and -d from centerline
        c1 = c - (d * sqrt(1 + (tan(yaw) ** 2)))
        c2 = c + (d * sqrt(1 + (tan(yaw) ** 2)))

        # Perpendicular distance of the vehicle from above to reference lines
        u_b = (a * x  + b * y + c1) / sqrt(a ** 2 + b ** 2)
        l_b = (a * x  + b * y + c2) / sqrt(a ** 2 + b ** 2) 

        # Compute constraint violation cost using the look up table computed in fit_curve function
        cost = self.lut_boundary_cost(u_b + l_b)

        return MX(vcat([cost, u_b + l_b]))

    def dist_to_ellipse(self, ref_state: np.ndarray, target_state: casadi.MX, a: float, b: float) -> (casadi.MX):
        """Computes the distance between two car centers with respected to an oriented and inflated ellipse
        
        Arguments
        ---------
        - ref_state: state of the car considered as a reference point to anchor the collision ellipse
        - target_state: state of the car which has to be checked for collison with the reference car
        - a: semi major axis of the collision ellipse
        - b: semi minor axis of the collision ellipse

        Returns
        -------
        - d > 1 if the target state is outside the reference ellipse (no collision)
        - d < 1 if the target state is inside the reference ellipse (collision)
        """
        x_c     = ref_state[0]
        y_c     = ref_state[1]
        theta   = -ref_state[2]

        x = target_state[0]
        y = target_state[1]

        d = ((((x_c - x) * cos(theta) - (y_c - y) * sin(theta)) ** 2) / (a ** 2)) + ((((x_c - x) * sin(theta) + (y_c - y) * cos(theta)) ** 2) / (b ** 2))

        return d

    def stanley_control(self, waypoints: np.ndarray, car_state: np.ndarray, max_steer_angle: float) -> (float):
        """Compute car steering command using the Stanley controller model assuming constant velocity
        
        Arguments
        ---------
        - waypoints: pre-computed waypoints to track
        - car_state: current state of the vehicle [x, y, yaw, vx, vy]
        - max_steer_angle: maximum steering angle of the spawned vehicle [degrees]

        Returns
        -------
        steer: steering command in carla sim range of [-1, 1]
        """
        x_des, y_des, yaw_des = waypoints[self.nearest_idx_c1 + 3]
        x, y, yaw, vx, vy = car_state

        d = np.sqrt((x - x_des) ** 2 + (y - y_des) ** 2)

        # Heading error [radians]
        psi_h = wrapToPi(yaw_des - yaw)

        # Crosstrack yaw difference to path yaw [radians]
        yaw_diff = wrapToPi(yaw_des - np.arctan2(y - y_des, x - x_des))

        # Crosstrack error in yaw [radians]
        psi_c = np.arctan2(0.5 * np.sign(yaw_diff) * d, 5.0 + vx)

        # Steering angle control
        steer = np.degrees(wrapToPi(psi_h + psi_c))
        steer = max(min(steer, max_steer_angle), -max_steer_angle) / max_steer_angle    # constrained to [-1, 1]

        return steer

    def mpc(self, ego_sols: dict, ego_state: np.ndarray, nearest_idx: int, opp_states: np.ndarray, system_params: np.ndarray, max_L: float, 
        L: np.ndarray, Ts: float, weights: dict, orient_flag: bool, plot_flag: bool, print_flag: bool) -> (dict):
        """Model Predictive Controller Function

        Arguments
        ---------
        - ego_sols: a dictionary containing all the solutions for the ego vehicle from the previous iteration of MPC
        - ego_state: current true state of the concerned vehicle
        - nearest_idx: nearest waypoint index to the current vehicle position
        - opp_states: opponent states for collision avoidance
        - system_params: Vehicle dynamics parameters obtained from System ID
        - max_L: maximum path length of the track
        - L: cumulative sum of distance between 2 consecutive waypoints to be used as a parameter for the splines
        - Ts: Sampling Time of the controller
        - weights: objective function weights
        - orient_flag: indicates the region where orientation jumps between[-pi] and [+pi] occur
        - plot_flag: Plot the path predictions output of the solver
        - print_flag: Debug print statements for solver output values

        Returns
        -------
        - ego_sols: updated dictionary containing all the solutions from the current iteration of MPC
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
            state_prediction = vertcat(x + sqrt((vx + d_nan) ** 2 + (vy + d_nan)** 2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                                y + sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                                fmod(((yaw + (sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * tan(delta) * dt / sqrt((l ** 2) + (0.5 * l * tan(delta)) ** 2))) + PI_2), PI_2),
                                vx + ((p * acc) - (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * vx - Cf * vx)) * dt,
                                vy - (Cc * (atan2(vy, vx + d_nan) - delta) + (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) + Cf) * vy) * dt)
        else:
            state_prediction = vertcat(x + sqrt((vx + d_nan) ** 2 + (vy + d_nan)** 2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                                y + sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                                (yaw + (sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * tan(delta) * dt / sqrt((l ** 2) + (0.5 * l * tan(delta)) ** 2))),
                                vx + ((p * acc) - (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * vx - Cf * vx)) * dt,
                                vy - (Cc * (atan2(vy, vx + d_nan) - delta) + (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) + Cf) * vy) * dt)

        pred = Function('pred', [state, control_command, dt], [state_prediction])

        prev_states = ego_sols['states']
        prev_controls = ego_sols['controls']
        prev_t = ego_sols['t']
        prev_v = ego_sols['v']

        # Set the weighting factors for different penalties/rewards in MPC
        w_gas       = weights['w_gas']                          # Throttle/brake penalty
        w_steer     = weights['w_steer']                        # Steer penalty
        w_lag       = weights['w_lag']                          # Lag Error penalty
        w_reward    = weights['w_reward']                       # Path progress reward
        w_roc       = weights['w_roc']                          # Rate of change of controls and velocity penalty 
        w_b         = weights['w_b']                            # Boundary constraints penalty 
        w_c         = weights['w_c']                            # Collision constraints penalty

        ##### Optistack begin #####
        opti = Opti()

        mpc_states = opti.variable(5, self.P + 1)       # States [x, y, yaw, vx, vy]
        lag_error = opti.variable(1, self.P + 1)        # Lag Error
        t = opti.variable(1, self.P + 1)                # Path length parameter
        v = opti.variable(1, self.P + 1)                # Path progression rate
        u = opti.variable(2, self.P)                    # Controls [throttle/brake, steering]
        track_boundary = opti.variable(1, self.P)       # Cost for approaching track boundaries
        centerline_offset = opti.variable(1, self.P)    # Change in lateral position of vehicle
        collision = opti.variable(1, self.P)            # bounding ellipse collision check

        # Penalty/Reward function to optimize over
        p_acc           = u[0, :] @ w_gas @ u[0, :].T                                   # Throttle/Brake magnitude
        p_steer         = u[1, :] @ w_steer @ u[1, :].T                                 # Steering magnitude
        p_lag           = lag_error @ w_lag @ lag_error.T                               # MPCC Lag error
        p_boundary      = track_boundary @ w_b @ track_boundary.T                       # Soft distance cost

        p_control_roc   = w_roc * sumsqr(u[:, :self.P-1] - u[:, 1:self.P])              # Rate of Change of Controls
        p_v_roc         = w_roc * sumsqr(v[:self.P] - v[1:self.P + 1])                  # Rate of Change of path velocity 
        p_wobble        = 1.5 * sumsqr(centerline_offset[1:] - centerline_offset[:-1])  # Wobble of path wrt centerline
        p_collision     = collision @ w_c @ collision.T                                 # Collision cost
        
        r_v_max         = ((v * Ts) @ w_reward @ (v * Ts).T)                            # Path Progression Reward

        # Minimization objective
        opti.minimize(p_acc + p_steer + p_lag + p_boundary + p_control_roc + p_v_roc + p_wobble + p_collision - r_v_max)
        
        # Constraints
        opti.subject_to(mpc_states[:, 0] == ego_state)          # Start prediction with true vehicle state
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
            opti.subject_to(centerline_offset[i] == self.track_constraints(mpc_states[:, i + 1], t[i + 1])[1])
            opti.subject_to(track_boundary[i] == self.track_constraints(mpc_states[:, i + 1], t[i + 1])[0])
            opti.subject_to(lag_error[i] == self.calculate_error(mpc_states[:, i], t[i]))
            opti.subject_to(t[i + 1] == t[i] + v[i] * dt)

            d = self.dist_to_ellipse(opp_states[:, i + 1], mpc_states[:, i + 1], 4.5, 3)
            opti.subject_to(collision[i] == self.lut_collision_cost(d))

        opti.subject_to(lag_error[-1] == self.calculate_error(mpc_states[:, -1], t[-1]))

        # Variable Initializations
        if orient_flag:
            opti.set_initial(mpc_states, np.vstack([ego_state] * (self.P + 1)).T)
        else:
            predicted_last_state = self.predict_new(orient_flag, prev_states[:, -1], prev_controls[:, -1], system_params, dt)
            opti.set_initial(mpc_states, np.vstack((ego_state, prev_states[:, 2:].T, predicted_last_state)).T)
        opti.set_initial(u, np.vstack((prev_controls[:, 1:].T, prev_controls[:, -1])).T)
        opti.set_initial(t, np.hstack((prev_t[1:], prev_t[-1] + prev_v[-1] * dt)))
        opti.set_initial(v, np.hstack((prev_v[1:], prev_v[-1])))
        opti.set_initial(track_boundary, 0)
        opti.set_initial(lag_error, 0)
        opti.set_initial(collision, 0)

        # Set ipopt options
        p_opts = {"print_time": False, 'ipopt.print_level': 0, "ipopt.expect_infeasible_problem": "yes", "ipopt.max_iter": 75}
        opti.solver('ipopt', p_opts)

        sol = opti.solve()

        opti_states = sol.value(mpc_states)
        opti_controls = sol.value(u)
        opti_errors = sol.value(lag_error)
        opti_t = sol.value(t)
        opti_v = sol.value(v)

        ego_sols['states'] = opti_states
        ego_sols['controls'] = opti_controls
        ego_sols['t'] = opti_t
        ego_sols['v'] = opti_v

        if plot_flag:
            # Plot predicted states
            plt.plot(opti_states[0, :], opti_states[1, :])
            plt.pause(0.01)
            plt.cla()

        if print_flag:
            print('Predicted states: \n {} \n'.format(opti_states[2, :]))
            print('Optimized Controls: \n {} \n'.format(opti_controls))
            print('Predicted parametewer: \n {} \n'.format(opti_t))
            print('Path progression: \n {} \n'.format(opti_v))
            print('Lag Error: \n {} \n'.format(opti_errors))

        return ego_sols

def main():
    try:
        # Initialize car environment
        env = CarEnv()

        # Load waypoints and car parameters obtained from System ID
        waypoints = read_file("../../Data/2D_waypoints.txt")
        sys_params = read_file('../../Data/params.txt')

        # Default params
        end_flag = [False, False]       # End of laps flag
        prev_idx = [0, 0]               # Previous waypoint index
        laps_completed = [0, 0]         # Laps completed

        # Read command line args
        argparser = argparse.ArgumentParser(description = __doc__)
        argparser.add_argument('-n', '--number-of-laps', default=1, type=int, help='number of laps desired in the race')
        argparser.add_argument('-s', '--save', action='store_true', help='Set to True to save the states and control data')
        argparser.add_argument('-f', '--filename', default='mpc_2_car', type=str, help='Filename for the data logs')
        argparser.add_argument('-p', '--plot', action='store_true', help='Set to True to plot the MPC predictions')
        argparser.add_argument('-d', '--print', action='store_true', help='Set to True to print the outputs from MPC loop')
        argparser.add_argument('--vmax1', default=35, type=float, help='Set maximum velocity for the vehicle 1')
        argparser.add_argument('--vmax2', default=35, type=float, help='Set maximum velocity for the vehicle 2')
        argparser.add_argument('--spawn-idx', default=0, type=int, help='Set the spawn index')
        args = argparser.parse_args()

        if args.save:
            states_filename_1 = '../../Data/GTC_vs_MPC/' + args.filename + '_car1_states.pickle'
            controls_filename_1 = '../../Data/GTC_vs_MPC/' + args.filename + '_car1_controls.pickle'
            states_file_1 = open(states_filename_1, 'wb')
            controls_file_1 = open(controls_filename_1, 'wb')

            states_filename_2 = '../../Data/GTC_vs_MPC/' + args.filename + '_car2_states.pickle'
            controls_filename_2 = '../../Data/GTC_vs_MPC/' + args.filename + '_car2_controls.pickle'
            states_file_2 = open(states_filename_2, 'wb')
            controls_file_2 = open(controls_filename_2, 'wb')


        # Spawn two vehicles at spawn_pose corresponding to spawn_idx index in waypoints list
        spawn_idx = args.spawn_idx
        env.nearest_idx_c1 = spawn_idx
        env.nearest_idx_c2 = spawn_idx
        vehicle_1, env.car_1_state, max_steer_angle_1 = env.spawn_vehicle_2D(spawn_idx, waypoints, -4)
        vehicle_2, env.car_2_state, max_steer_angle_2 = env.spawn_vehicle_2D(spawn_idx, waypoints, 4)

        if args.save:
            cpickle.dump(np.r_[env.car_1_state], states_file_1)
            cpickle.dump(np.r_[0.0, 0.0, 0.0], controls_file_1)
            cpickle.dump(np.r_[env.car_2_state], states_file_2)
            cpickle.dump(np.r_[0.0, 0.0, 0.0], controls_file_2)

        # Spawn time car stabilization
        for i in range(50):
            env.world.tick()

        # Extended list of waypoints and distance to each waypoint from starting position
        waypoints_ext = np.vstack((waypoints, waypoints[:50, :]))
        l = np.cumsum(np.sqrt(np.sum(np.square(waypoints_ext[:-1, :2] - waypoints_ext[1:, :2]), axis = 1)))
        L = np.r_[0, l]
        max_L = L[waypoints.shape[0] + 1]
        
        # Fit 3 degree b-splines for the waypoint poses and boundary collision penalty
        env.lut_x, env.lut_y, env.lut_theta, env.lut_boundary_cost, env.lut_collision_cost = env.fit_curve(waypoints_ext, L)

        # Set MPC optimization variables' penalty weights
        env.set_mpc_params(P = 25)
        weights1 = {}
        weights1['w_gas']       = env.w_matrix(1, 0)[:-1, :-1]
        weights1['w_steer']     = env.w_matrix(1, 0)[:-1, :-1]
        weights1['w_lag']       = env.w_matrix(3, -0.02)
        weights1['w_reward']    = env.w_matrix(8, -0.2)
        weights1['w_b']         = env.w_matrix(init_value=1, step_size=1)[:-1, :-1]
        weights1['w_roc']       = 5
        weights1['w_c']         = env.w_matrix(init_value=10, step_size=0)[:-1, :-1]

        env.set_mpc_params(P = 25)
        weights2 = {}
        weights2['w_gas']       = env.w_matrix(1, 0)[:-1, :-1]
        weights2['w_steer']     = env.w_matrix(1, 0)[:-1, :-1]
        weights2['w_lag']       = env.w_matrix(3, -0.02)
        weights2['w_reward']    = env.w_matrix(8, -0.2)
        weights2['w_b']         = env.w_matrix(init_value=1, step_size=1)[:-1, :-1]
        weights2['w_roc']       = 5
        weights2['w_c']         = env.w_matrix(init_value=10, step_size=0)[:-1, :-1]
        
        # Controller and Prediction time step          
        Ts = 0.1

        prev_states_1 = np.vstack([env.car_1_state] * (env.P + 1)).T
        prev_controls_1 = np.zeros((2, env.P))
        prev_controls_1[0, :] = prev_controls_1[0, :] + 0.5
        prev_t_1 = np.ones(env.P + 1) * env.nearest_idx_c1
        prev_v_1 = np.zeros(env.P + 1)

        prev_sols_1 = {}
        prev_sols_1['states'] = prev_states_1
        prev_sols_1['controls'] = prev_controls_1
        prev_sols_1['t'] = prev_t_1
        prev_sols_1['v'] = prev_v_1

        prev_states_2 = np.vstack([env.car_2_state] * (env.P + 1)).T
        prev_controls_2 = np.zeros((2, env.P))
        prev_controls_2[0, :] = prev_controls_2[0, :] + 0.5
        prev_t_2 = np.ones(env.P + 1) * env.nearest_idx_c2
        prev_v_2 = np.zeros(env.P + 1)

        prev_sols_2 = {}
        prev_sols_2['states'] = prev_states_2
        prev_sols_2['controls'] = prev_controls_2
        prev_sols_2['t'] = prev_t_2
        prev_sols_2['v'] = prev_v_2

        total_iterations = 0
        fallbacks_1 = 0
        fallbacks_2 = 0

        # Initialize control loop        
        while(1):
            total_iterations += 1
            if ((total_iterations % 50) == 0):
                print('Success rate of GTC car 1 = ', (total_iterations - fallbacks_1) / total_iterations)
                print('Success rate of MPC car 2 = ', (total_iterations - fallbacks_2) / total_iterations)

            if (1500 > env.nearest_idx_c1 > 1250):
                orient_flag_1 = True
            else:
                orient_flag_1 = False
            
            if (1500 > env.nearest_idx_c2 > 1250):
                orient_flag_2 = True
            else:
                orient_flag_2 = False

            env.car_1_state    = env.get_true_state(vehicle_1, orient_flag_1)
            env.nearest_idx_c1 = env.calculate_nearest_index(env.car_1_state, waypoints)
            env.car_2_state    = env.get_true_state(vehicle_2, orient_flag_2)
            env.nearest_idx_c2 = env.calculate_nearest_index(env.car_2_state, waypoints)

            # env.cam_sensor.listen(lambda image: image.save_to_disk('/home/dhagash/MS-GE-02/MSR-Project/camera_pos_fix/%06d.png' % image.frame))
            
            # GTC First Car
            ibr_iters = 1
            for i in range(ibr_iters):
                if not end_flag[0]:
                    try:
                        # Set controller tuning params
                        env.set_mpc_params(P = 25, vmax = args.vmax1)
                        prev_sols_1 = env.mpc(prev_sols_1, env.car_1_state, env.nearest_idx_c1, prev_sols_2,
                         sys_params, max_L, L, Ts, weights1, orient_flag_1, args.plot, args.print)
                    
                        steer_1 = prev_sols_1['controls'][1, 0] / 1.22

                    except RuntimeError as err:
                        if i < ibr_iters - 1:
                            pass
                        else:
                            fallbacks_1 += 1
                            print('Car 1: Error occured with following error message: \n {} \n'.format(err))
                            print('Car 1: Fallback on Stanley Control !!!')
                            steer_1 = env.stanley_control(waypoints, env.car_1_state, max_steer_angle_1)
                            
                            prev_sols_1['controls'][0, :] = prev_sols_1['controls'][0, 0]
                            prev_sols_1['controls'][1, :] = steer_1
                            prev_sols_1['states'] = np.vstack([env.car_1_state] * (env.P + 1)).T
                            prev_sols_1['t'] = prev_sols_1['t'] + env.car_1_state[3] * Ts
                            pass
                    
                    if prev_controls_1[0, 0] > 0:
                        throttle_1 = prev_sols_1['controls'][0, 0]
                        brake_1 = 0
                    else:
                        brake_1 = prev_sols_1['controls'][0, 0]
                        throttle_1 = 0
            
            # MPC Second car
            if not end_flag[1]:
                try:
                    # Set controller tuning params
                    env.set_mpc_params(P = 25, vmax = args.vmax2)
                    obst_states = env.const_vel_model(env.car_1_state, Ts, env.P + 1)
                    prev_sols_2 = env.mpc(prev_sols_2, env.car_2_state, env.nearest_idx_c2, obst_states,
                     sys_params, max_L, L, Ts, weights2, orient_flag_2, args.plot, args.print)
                    
                    steer_2 = prev_sols_2['controls'][1, 0] / 1.22

                except RuntimeError as err:
                    fallbacks_2 += 1
                    print('Car 2: Error occured with following error message: \n {} \n'.format(err))
                    print('Car 2: Fallback on Stanley Control !!!')
                    steer_2 = env.stanley_control(waypoints, env.car_2_state, max_steer_angle_2)

                    prev_sols_2['controls'][0, :] = prev_sols_2['controls'][0, 0]
                    prev_sols_2['controls'][1, :] = steer_2
                    prev_sols_2['states'] = np.vstack([env.car_2_state] * (env.P + 1)).T
                    prev_sols_2['t'] = prev_sols_2['t'] + env.car_2_state[3] * Ts
                    pass
            
                if prev_sols_2['controls'][0, 0] > 0:
                        throttle_2 = prev_sols_2['controls'][0, 0]
                        brake_2 = 0
                else:
                    brake_2 = prev_sols_2['controls'][0, 0]
                    throttle_2 = 0

            if args.plot:
                plt.subplot(2, 1, 1)
                plt.plot(prev_sols_1['states'][0, :], prev_sols_1['states'][1, :], 'r')
                plt.subplot(2, 1, 2)
                plt.plot(prev_sols_2['states'][0, :], prev_sols_2['states'][1, :], 'b')  
                plt.pause(0.01)
            
            if (env.nearest_idx_c1 == waypoints.shape[0] - 1) and env.nearest_idx_c1 != prev_idx[0]:
                laps_completed[0] += 1
                prev_sols_1['t'] = prev_sols_1['t'] - prev_sols_1['t'][0]
                if laps_completed[0] == args.number_of_laps:
                    end_flag[0] = True
                    throttle_1 = 0
                    steer_1 = 0
                    brake_1 = 0.2

            prev_idx[0] = env.nearest_idx_c1

            if (env.nearest_idx_c2 == waypoints.shape[0] - 1) and env.nearest_idx_c2 != prev_idx[1]:
                laps_completed[1] += 1
                prev_sols_2['t'] = prev_sols_2['t'] - prev_sols_2['t'][0]
                if laps_completed[1] == args.number_of_laps:
                    end_flag[1] = True
                    throttle_2 = 0
                    steer_2 = 0
                    brake_2 = 0.2

            prev_idx[1] = env.nearest_idx_c2

            prev_idx[1] = env.nearest_idx_c2

            vehicle_1.apply_control(carla.VehicleControl(throttle = throttle_1, steer = steer_1, reverse = False, brake = brake_1))
            vehicle_2.apply_control(carla.VehicleControl(throttle = throttle_2, steer = steer_2, reverse = False, brake = brake_2))
            env.world.tick()

            if args.save:
                cpickle.dump(env.car_1_state, states_file_1)
                cpickle.dump(np.r_[throttle_1, brake_1, steer_1], controls_file_1)

                cpickle.dump(env.car_2_state, states_file_2)
                cpickle.dump(np.r_[throttle_2, brake_2, steer_2], controls_file_2)

            if env.car_1_state[3] == 0.0 and env.car_2_state[3] == 0 and laps_completed[0] == args.number_of_laps and laps_completed[1] == args.number_of_laps:
                break

    finally:
        if args.save:
            states_file_1.close()
            controls_file_1.close()

            states_file_2.close()
            controls_file_2.close()

        # Destroy all actors in the simulation
        env.destroy()

if __name__=="__main__":
    main()