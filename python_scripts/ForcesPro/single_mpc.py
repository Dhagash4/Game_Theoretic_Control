#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

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

import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('/home/saurabh/Forcespro')

import forcespro
from forcespro.modelling import InterpolationFit
import forcespro.nlp

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
        self.settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(self.settings)

        # Get Map of the world
        self.map = self.world.get_map()
        
        # Get all the blueprints available in the world
        self.blueprint_library =  self.world.get_blueprint_library()
        
        # Initialize state dataclass list to log subsequent state information
        self.states = []

        # Initialize control dataclass list to log subsequent control commands
        self.controls = []

        # Nearest Waypoints Index
        self.nearest_idx = 0

        # Vehicle Current Pose
        self.car_state = np.empty(5)

        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []

        # Control Command
        self.control = np.array([0, 0])

    def spawn_vehicle_2D(self, spawn_idx: int, waypoints: np.ndarray, offset: float) -> Tuple[carla.libcarla.Vehicle, np.ndarray, float]:
        """Spawn a Vehicle at given index among the list of waypoints

        Arguments
        ---------
        - spawn_idx: spawn index from the waypoints list
        - waypoints: pre-computed waypoints to track
        - offset: offset distance (signed) from centerline

        Returns
        -------
        - vehicle: carla instance to the spawned vehicle
        - spawn_state: 5 parameter state of the vehicle [x, y, yaw, vx, vy]
        - max_steer_angle: maximum steering angle of the spawned vehicle [degrees]
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
        """Attach a camera to the vehicle for recording images from car POV
        """
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

        x_ref = self.lut_x(theta)
        y_ref = self.lut_y(theta)
        yaw = self.lut_theta(theta)

        e_lag = -cos(yaw) * (x - x_ref) - sin(yaw) * (y - y_ref)

        return e_lag

    def continuous_dynamics(self, state, control):
        """Defines dynamics of the car, i.e. equality constraints.
        
        Arguments
        ---------
        - state: state of the vehicle [x, y, yaw, vx, vy]
        - control: control commands given to the vehicle [throttle/brake, steer]

        Returns
        -------
        - derivative: rate of change of the vehicle states for given control commands
        """
        l, p, Cd, Cf, Cc = 4.46, 4.22, -0.01, 0.36, 1.32
        d_nan = 1e-5 # Add small number to avoid NaN error at zero velocities in the Jacobian evaluation
        dt = 0.1

        x, y, yaw, vx, vy = state[0], state[1], state[2], state[3], state[4]
        lag_error, t, v, d_s, c = state[5], state[6], state[7], state[8], state[9]
        
        acc, delta = control[0], control[1]

        new_state = vertcat(x + sqrt((vx + d_nan) ** 2 + (vy + d_nan)** 2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                            y + sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                            (yaw + (sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * tan(delta) * dt / sqrt((l ** 2) + (0.5 * l * tan(delta)) ** 2))),
                            vx + ((p * acc) - (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) * vx - Cf * vx)) * dt,
                            vy - (Cc * (atan2(vy, vx + d_nan) - delta) + (Cd * sqrt((vx + d_nan) ** 2 + (vy + d_nan) ** 2) + Cf) * vy) * dt)
        new_t = t + v * dt

        final_state = vertcat(new_state, self.calculate_error(new_state, new_t), new_t, self.track_constraints(new_state, new_t)[0], self.track_constraints(new_state, new_t)[1])
        
        return final_state

    def generate_pathplanner(self, max_L):

        # Model Definition
        model = forcespro.nlp.SymbolicModel()
        model.N = self.P  # horizon length
        model.nvar = 12  # number of variables
        model.neq = 9  # number of equality constraints
        model.nh = 0  # number of inequality constraint functions
        model.npar = 5 # number of runtime parameters

        model.objective = lambda z, p: self.opt_objective(z, p)

        model.eq = lambda z: self.continuous_dynamics(z[2:], z[:2])

        model.E = np.concatenate([np.zeros((9, 2)), np.eye(9, 7), np.zeros((9, 1)), np.eye(9, 2, -7)], axis=1)
        
        model.lb = np.array([-1.0, -1.22, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf])
        model.ub = np.array([1.0, 1.22, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max_L + 50, self.vmax, np.inf, np.inf])
        
        model.xinitidx = [2, 3, 4, 5, 6, 8]
        
        # Solver generation
        # -----------------

        # Set solver options
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
        codeoptions.maxit = 400     # Maximum number of iterations
        codeoptions.printlevel = 0  
        codeoptions.optlevel = 2    # 0 no optimization, 1 optimize for size, 
        #                             2 optimize for speed, 3 optimize for size & speed
        codeoptions.nlp.bfgs_init = 3.0*np.identity(12) # initialization of the hessian
        #                             approximation
        codeoptions.noVariableElimination = 1.
        # change this to your server or leave uncommented for using the 
        # standard embotech server at https://forces.embotech.com 
        # codeoptions.server = 'https://forces.embotech.com'
        
        # Creates code for symbolic model formulation given above, then contacts 
        # server to generate new solver
        solver = model.generate_solver(options=codeoptions)

        return model, solver

    def opt_objective(self, z, p):
        obj = (p[0] * z[0] ** 2) + (p[1] * z[1] ** 2) - (p[2] * (z[9] * 0.1) ** 2) + (p[3] * z[7] ** 2) + (p[4] * z[10])
        return obj

    def set_mpc_params(self, P: int, vmax: float) -> NoReturn:
        """Set parameters needed for the Model Predictive controller
        
        Arguments
        ---------
        - P: Prediction horizon of the controller
        - vmax: maximum allowed velocity for the vehicle
        """
        self.P = P
        self.vmax = vmax
        
    def set_opti_weights(self, weights: dict) -> NoReturn:
        """Set the weighting factors for different penalties/rewards in MPC
        """
        self.w_u0 = weights['w_u0']         # Throttle/brake penalty
        self.w_u1 = weights['w_u1']         # Steer penalty
        self.w_lag = weights['w_lag']       # Lag Error penalty
        self.gamma = weights['gamma']       # Path progress reward
        self.w_ds = weights['w_ds']         # Boundary constraints penalty 

    def w_matrix(self, init_value: float, step_size: float) -> (np.ndarray):
        w = np.zeros(self.P + 1)
        for i in range(self.P):
            w[i] = init_value + step_size * i
        return w

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
        - lut_d: Look up table for the track boundary constraint costs
        - lut_collide: Look up table for the collision constraint costs
        """
        # Waypoints interpolation
        self.lut_x = InterpolationFit(L, waypoints[:, 0])
        self.lut_y = InterpolationFit(L, waypoints[:, 1])
        self.lut_theta = InterpolationFit(L, waypoints[:, 2])

        # Soft constraint cost for track boundaries
        t = 6                               # Threshold
        cost_fit = np.zeros((10000))
        numbers = np.linspace(-16,16,10000)
        for i in range(10000):
            if -t <= numbers[i] <= t:
                cost_fit[i] = 0.0
            else:
                cost_fit[i] = (abs(numbers[i]) - t) ** 2
        self.lut_d = InterpolationFit(numbers, cost_fit)

        return self.lut_x, self.lut_y, self.lut_theta, self.lut_d

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
        x_ref = self.lut_x(theta)
        y_ref = self.lut_y(theta)
        yaw = self.lut_theta(theta)
        
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
        cost = self.lut_d(u_b + l_b)

        return vcat([cost, u_b + l_b])

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

def main():
    try:
        # Initialize car environment
        env = CarEnv()

        # Load waypoints and car parameters obtained from System ID
        waypoints = read_file("../../Data/2D_waypoints.txt")
        sys_params = read_file('../../Data/params.txt')

        # Default params
        end_flag = False       # End of laps flag
        prev_idx = 0               # Previous waypoint index
        laps_completed = 0         # Laps completed

        # Read command line args
        argparser = argparse.ArgumentParser(description = __doc__)
        argparser.add_argument('-n', '--number-of-laps', default=1, type=int, help='number of laps desired in the race')
        argparser.add_argument('-s', '--save', action='store_true', help='Set to True to save the states and control data')
        argparser.add_argument('-p', '--plot', action='store_true', help='Set to True to plot the MPC predictions')
        argparser.add_argument('-d', '--print', action='store_true', help='Set to True to print the outputs from MPC loop')
        argparser.add_argument('--vmax', default=35, type=float, help='Set maximum velocity for the vehicle 1')
        argparser.add_argument('--spawn-idx', default=0, type=int, help='Set the spawn index')
        args = argparser.parse_args()

        # Spawn two vehicles at spawn_pose corresponding to spawn_idx index in waypoints list
        spawn_idx = args.spawn_idx
        env.nearest_idx = spawn_idx
        vehicle, env.car_state, max_steer_angle = env.spawn_vehicle_2D(spawn_idx, waypoints, 0)

        if args.save:
            env.states.append(state(0, env.car_state[0], env.car_state[1], env.car_state[2], env.car_state[3], env.car_state[4], 0))
            env.controls.append(control(0, 0.0, 0.0, 0.0, 0.0, 0))

        # Spawn time car stabilization
        for i in range(50):
            env.world.tick()

        # Extended list of waypoints and distance to each waypoint from starting position
        waypoints_ext = np.vstack((waypoints, waypoints[:50, :]))
        l = np.cumsum(np.sqrt(np.sum(np.square(waypoints_ext[:-1, :2] - waypoints_ext[1:, :2]), axis = 1)))
        L = np.r_[0, l]
        max_L = L[waypoints.shape[0] + 1]
        
        # Fit 3 degree b-splines for the waypoint poses and boundary collision penalty
        env.lut_x, env.lut_y, env.lut_theta, env.lut_d = env.fit_curve(waypoints_ext, L)

        env.set_mpc_params(P = 25, vmax = args.vmax)

        model, solver = env.generate_pathplanner(max_L)
        print('solver generated')
        
        # Set MPC optimization variables' penalty weights
        weights = {}
        weights['w_u0'] = env.w_matrix(1, 0)[:-1, :-1]
        weights['w_u1'] = env.w_matrix(2, 0)[:-1, :-1]
        weights['w_lag'] = env.w_matrix(3, -0.02)
        weights['gamma'] = env.w_matrix(8, -0.2)
        weights['w_ds'] = env.w_matrix(init_value=1, step_size=1)[:-1, :-1]

        env.set_opti_weights(weights)               
        Ts = 0.1

        prev_states = np.vstack([env.car_state] * (env.P)).T
        prev_controls = np.zeros((2, env.P))
        prev_t = np.ones(env.P) * env.nearest_idx
        prev_v = np.zeros(env.P)

        prev_vals = {}
        prev_vals['states'] = prev_states
        prev_vals['controls'] = prev_controls
        prev_vals['t'] = prev_t
        prev_vals['v'] = prev_v

        total_iterations = 0
        fallbacks = 0

        params = np.zeros((env.P, 5))
        params[:, 0] = weights['w_u0']
        params[:, 1] = weights['w_u1']
        params[:, 2] = weights['gamma']
        params[:, 3] = weights['w_lag']
        params[:, 4] = weights['w_ds']
        params = params.reshape(1, -1)

        # Initialize control loop
        while(1):

            x0 = np.hstack((prev_vals['controls'].T, prev_vals['states'], np.zeros((env.P, 1)), prev_vals['t'].reshape(-1, 1), prev_vals['v'].reshape(-1, 1), np.zeros((env.P, 2)))).reshape(1, -1)
            xinit = np.r_[env.car_state, env.nearest_idx]

            problem = {"x0": x0, "xinit": xinit.T}
            problem["all_parameters"] = params.T

            total_iterations += 1
            if ((total_iterations % 50) == 0):
                print('Success rate of MPC car 1 = ', (total_iterations - fallbacks) / total_iterations)

            if (1500 > env.nearest_idx > 1250):
                orient_flag = True
            else:
                orient_flag = False

            env.car_state = env.get_true_state(vehicle, orient_flag)
            env.nearest_idx = env.calculate_nearest_index(env.car_state, waypoints)

            print(env.car_state[3])

            # env.cam_sensor.listen(lambda image: image.save_to_disk('/home/dhagash/MS-GE-02/MSR-Project/camera_pos_fix/%06d.png' % image.frame))

            if not end_flag:
                try:
                    output, exitflag, info = solver.solve(problem)
                    print(output)

                    steer = prev_vals['controls'][1, 0] / 1.22

                    if prev_vals['controls'][0, 0] > 0:
                        throttle = prev_vals['controls'][0, 0]
                        brake = 0
                    else:
                        brake = prev_vals['controls'][0, 0]
                        throttle = 0

                except RuntimeError as err:
                    fallbacks += 1
                    print('Car 1: Error occured with following error message: \n {} \n'.format(err))
                    print('Car 1: Fallback on Stanley Control !!!')
                    steer = env.stanley_control(waypoints, env.car_state, max_steer_angle)

                    if prev_controls[0, 0] > 0:
                        throttle = prev_vals['controls'][0, 0]
                        brake = 0
                    else:
                        brake = prev_vals['controls'][0, 0]
                        throttle = 0              

                    prev_vals['controls'][0, :] = prev_vals['controls'][0, 0]
                    prev_vals['controls'][1, :] = steer
                    prev_vals['states'] = np.vstack([env.car_state] * (env.P + 1)).T
                    prev_vals['t'] = prev_vals['t'] + env.car_state[3] * Ts
                    pass
            
            
            if (env.nearest_idx == waypoints.shape[0] - 1) and env.nearest_idx != prev_idx:
                laps_completed += 1
                prev_vals['t'] = prev_vals['t'] - prev_vals['t'][0]
                if laps_completed == args.number_of_laps:
                    end_flag = True
                    throttle = 0
                    steer = 0
                    brake = 0.2

            prev_idx = env.nearest_idx

            vehicle.apply_control(carla.VehicleControl(throttle = throttle, steer = steer, reverse = False, brake = brake))
            env.world.tick()

            if args.save:
                env.states.append(state(0, env.car_state[0], env.car_state[1], env.car_state[2], env.car_state[3], env.car_state[4], 0))
                env.controls.append(control(0, throttle, brake, env.control[0], steer, 0))

            if env.car_state[3] == 0.0 and laps_completed == args.number_of_laps:
                break

    finally:
        if args.save:
        # Save all Dataclass object lists to a file
            save_log('../../Data/states_mpc.pickle', env.states)
            save_log('../../Data/controls_mpc.pickle', env.controls)

        # Destroy all actors in the simulation
        env.destroy()

if __name__=="__main__":
    main()