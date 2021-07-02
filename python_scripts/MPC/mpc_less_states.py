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
        self.states = []

        # Initialize control dataclass list to log subsequent control commands
        self.controls = []

        # Track errors
        self.errors = []

        # Nearest Waypoints Index
        self.nearest_wp_idx = 0

        # Vehicle Current Pose
        self.car_state = np.empty(4)

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
        spawn_state = np.r_[waypoints[spawn_idx], 0]
        spawn_state[0] = spawn_state[0] - 0 * np.sin(spawn_state[2])
        spawn_state[1] = spawn_state[1] + 0 * np.cos(spawn_state[2])

        spawn_tf = carla.Transform(carla.Location(spawn_state[0], spawn_state[1], 2), carla.Rotation(0, np.degrees(spawn_state[2]), 0))
        self.vehicle =  self.world.spawn_actor(car_model, spawn_tf)

        # Get max steering angle of car's front wheels
        self.max_steer_angle = self.vehicle.get_physics_control().wheels[0].max_steer_angle
        
        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []
        # Append our vehicle to actor list
        self.actor_list.append(self.vehicle)

        # Add spawn state to the states data log
        self.states.append(state(0, spawn_state[0], spawn_state[1], spawn_state[2], spawn_state[3], 0, 0))

        return spawn_state


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


    def get_true_state(self, orient_flag) -> (np.ndarray):
        """Get vehicles true state from the simulation
        
        Returns
        -------
        - true_state: True state of the vehicle [x, y, yaw, vx, vy]
        """
        x = self.vehicle.get_transform().location.x
        y = self.vehicle.get_transform().location.y
        if orient_flag:
            yaw = ((np.radians(self.vehicle.get_transform().rotation.yaw)) + PI_2) % PI_2    # [radians]
        else:
            yaw = wrapToPi(np.radians(self.vehicle.get_transform().rotation.yaw))    # [radians]
        vx = self.vehicle.get_velocity().x * np.cos(yaw) + self.vehicle.get_velocity().y * np.sin(yaw)
        vy = -self.vehicle.get_velocity().x * np.sin(yaw) + self.vehicle.get_velocity().y * np.cos(yaw)
        v = np.sqrt(vx ** 2 + vy ** 2)

        true_state = np.array([x, y, yaw, v])

        return true_state


    def predict_new(self, orient_flag, old_state: np.ndarray, control: np.ndarray, params: np.ndarray, dt: float) -> (np.ndarray):
        L, p, Cd, Cf, Cc = params

        x, y, theta, v = old_state

        acc, delta = control

        x_new = x + (v * np.cos(theta) * dt)
        y_new = y + (v * np.sin(theta) * dt)
        if orient_flag:
            theta_new = (theta + (v * np.tan(delta) * dt / L + PI_2)) % PI_2
        else:
            theta_new = wrapToPi(theta + (v * np.tan(delta) * dt / L))
        v_new = v + p * acc * dt

        new_state = np.array([x_new, y_new, theta_new, v_new])

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

    def calculate_error(self, state_mpc: MX, theta: MX, max_L: float) -> (MX):
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
        self.theta = 0

    def set_opti_weights(self, w_u0: float, w_u1: float, w_ql: float, gamma: float, w_c: float, w_ds: float) -> NoReturn:
        self.w_u0 = w_u0
        self.w_u1 = w_u1
        self.w_ql = w_ql
        self.gamma = gamma
        self.w_c = w_c
        self.w_ds = w_ds

    def w_matrix(self,init_value,step_size):
        w = np.eye(self.P + 1)
        for i in range(self.P + 1):
            w[i, i] = init_value + step_size * i
        return w

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
        # Waypoints interpolation
        waypoints_ext = np.vstack((waypoints, waypoints[:50, :]))
        L = np.arange(0, waypoints_ext.shape[0])

        self.lut_x = interpolant('LUT_x', 'bspline', [L], waypoints_ext[:, 0], dict(degree=[3]))
        self.lut_y = interpolant('LUT_y', 'bspline', [L], waypoints_ext[:, 1], dict(degree=[3]))
        self.lut_theta = interpolant('LUT_t', 'bspline', [L], waypoints_ext[:, 2], dict(degree=[3]))

        # Soft constraint cost for track boundaries
        cost_fit = np.zeros((10000))
        numbers = np.linspace(-16,16,10000)
        for i in range(10000):
            cost_fit[i] = self.cost_track_const(numbers[i])
        self.lut_d = interpolant('LUT_d', 'bspline', [numbers], cost_fit, dict(degree=[3]))

        return self.lut_x, self.lut_y, self.lut_theta, self.lut_d
    
    def cost_track_const(self, d: float):
        b = 2
        if -b <= d <= b:
            cost = 0.0
        else:
            cost = (abs(d) - b) ** 2
        return cost

    def track_constraints(self, state_mpc: MX, theta: MX) -> (MX): 
        x = state_mpc[0]
        y = state_mpc[1]

        theta = theta
        x_ref = self.lut_x(theta)
        y_ref = self.lut_y(theta)
        yaw = self.lut_theta(theta)
        
        track_width = 8.0           # [m]
        d = (track_width * 0.75)/2
        
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


    def stanley_control(self, waypoints):
        x_des, y_des, yaw_des = waypoints[self.nearest_wp_idx + 3]
        x, y, yaw, v = self.car_state

        d = np.sqrt((x - x_des) ** 2 + (y - y_des) ** 2)

        # Heading error [radians]
        psi_h = wrapToPi(yaw_des - yaw)

        # Crosstrack yaw difference to path yaw [radians]
        yaw_diff = wrapToPi(yaw_des - np.arctan2(y - y_des, x - x_des))

        # Crosstrack error in yaw [radians]
        psi_c = np.arctan2(2 * np.sign(yaw_diff) * d, 5.0 + v)

        # Steering angle control
        steer = np.degrees(wrapToPi(psi_h + psi_c))     # uncontrained in degrees
        steer = max(min(steer, self.max_steer_angle), -self.max_steer_angle)
        steer = (steer)/self.max_steer_angle            # constrained to [-1, 1]

        return steer


    def mpc(self, orient_flag, prev_states, prev_controls, prev_t, prev_v, system_params: np.ndarray, max_L: float, Ts: float):
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
        L, p, Cd, Cf, Cc = system_params
        d_nan = 1e-5 # Add small number to avoid NaN error at zero velocities in the Jacobian evaluation

        dt = MX.sym('dt')
        state = MX.sym('s', 4)
        control_command = MX.sym('u', 2)

        x, y, yaw, v = state[0], state[1], state[2], state[3]
        acc, delta = [control_command[0], control_command[1]]

        if orient_flag:
            state_prediction = vertcat(x + v * cos(yaw) * dt,
                                y + v * sin(yaw) * dt,
                                fmod((yaw + (v / L) * tan(delta) * dt + PI_2), PI_2),
                                v + p * acc * dt)
        else:
            state_prediction = vertcat(x + v * cos(yaw) * dt,
                                y + v * sin(yaw) * dt,
                                yaw + (v / L) * tan(delta) * dt,
                                v + p * acc * dt)

        pred = Function('pred', [state, control_command, dt], [state_prediction])

        ##### Optistack begin #####
        opti = Opti()

        s = opti.variable(4, self.P + 1)    # States [x, y, yaw, v]
        e = opti.variable(1, self.P + 1)    # Lag Error
        t = opti.variable(1, self.P + 1)    # Path length parameter
        v = opti.variable(1, self.P + 1)    # Path progression rate
        u = opti.variable(2, self.P)        # Controls [throttle/brake, steering]
        d_s = opti.variable(1, self.P)
        c = opti.variable(1, self.P)

        # Costs to optimize over
        p_acc   =  u[0, :]  @ self.w_u0 @ u[0, :].T                             # Throttle/Brake cost
        p_steer =  u[1, :] @  self.w_u1 @ u[1, :].T                             # Steering cost
        p_control_roc = self.w_c * sumsqr(u[:, :self.P-1] - u[:, 1:self.P])     # Rate of Change of Controls

        p_v_roc = self.w_c * sumsqr(v[:self.P] - v[1:self.P + 1])               # Rate of Change of progression rate 
        r_v_max = ((v* Ts) @ self.gamma @ (v* Ts).T)                            # Progression Reward
        p_lag = e @ self.w_ql @ e.T                                             # Lag error
        c_d_s  = d_s @ self.w_ds @ d_s.T 

        # Minimization objective
        opti.minimize(p_lag + p_control_roc + p_v_roc + p_acc + p_steer - r_v_max + 0.5 * sumsqr(c[1:] - c[:-1] + c_d_s)
                        )
        
        # Constraints
        opti.subject_to(s[:, 0] == self.car_state)              # Start prediction with true vehicle state
        opti.subject_to(t[0] == self.nearest_wp_idx)                     

        opti.subject_to(opti.bounded(-1.0, u[0, :], 1.0))       # Bounded steering
        opti.subject_to(opti.bounded(-1.22, u[1, :], 1.22))     # Bounded throttle/brake
        opti.subject_to(opti.bounded(0, t, max_L + 50))         # Bounded path length
        opti.subject_to(opti.bounded(0, v, self.vmax))          # Bounded path progression
        
        # Prediction horizon
        for i in range(self.P):
            if i < 0.6 * self.P:
                dt = Ts
            else:
                dt = Ts + 0

            opti.subject_to(s[:, i+1] == pred(s[:, i], u[:, i], dt))
            opti.subject_to(c[i] == self.track_constraints(s[:, i + 1], t[i + 1])[1])
            opti.subject_to(d_s[i] == self.track_constraints(s[:, i + 1], t[i + 1])[0])
            opti.subject_to(e[i] == self.calculate_error(s[:, i], t[i], max_L))
            opti.subject_to(t[i + 1] == t[i] + v[i] * dt)

        opti.subject_to(e[-1] == self.calculate_error(s[:, -1], t[-1], max_L))

        # Variable Initializations
        if orient_flag:
            opti.set_initial(s, np.vstack([self.car_state] * (self.P + 1)).T)
        else:
            predicted_last_state = self.predict_new(orient_flag, prev_states[:, -1], prev_controls[:, -1], system_params, dt)
            opti.set_initial(s, np.vstack((self.car_state, prev_states[:, 2:].T, predicted_last_state)).T)
        opti.set_initial(u, np.vstack((prev_controls[:, 1:].T, prev_controls[:, -1])).T)
        opti.set_initial(t, np.hstack((prev_t[1:], prev_t[-1] + prev_v[-1] * dt)))
        opti.set_initial(v, np.hstack((prev_v[1:], prev_v[-1])))
        opti.set_initial(d_s, 0)
        opti.set_initial(e, 0)

        # Set ipopt options
        p_opts = {"print_time": False, 'ipopt.print_level': 0, "ipopt.expect_infeasible_problem": "yes", "ipopt.max_iter": 75}
        opti.solver('ipopt', p_opts)

        sol = opti.solve()

        opti_states = sol.value(s)
        opti_controls = sol.value(u)
        opti_errors = sol.value(e)
        opti_t = sol.value(t)
        opti_v = sol.value(v)

        # Plot predicted states
        # plt.plot(opti_states[1, :], opti_states[0, :])
        # plt.pause(0.1)
        # plt.cla()

        # print('Predicted states: \n {} \n'.format(opti_states[2, :]))
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
        env.set_mpc_params(P = 25, vmax = 35)
        env.set_opti_weights(w_u0 = env.w_matrix(1, 0)[:-1, :-1], w_u1 = env.w_matrix(2, 0)[:-1, :-1], w_ql = env.w_matrix(3, -0.02), 
                            gamma = env.w_matrix(8, -0.2), w_c = 5, w_ds = env.w_matrix(init_value=1, step_size=1)[:-1, :-1])
        Ts = 0.1

        prev_states = np.vstack([env.car_state] * (env.P + 1)).T
        prev_controls = np.zeros((2, env.P))
        prev_controls[0, :] = prev_controls[0, :] + 0.5
        prev_t = np.ones(env.P + 1) * env.nearest_wp_idx
        prev_v = np.zeros(env.P + 1)

        # Initialize control loop
        while(1):
            if (1500 > env.nearest_wp_idx > 1250):
                orient_flag = True
            else:
                orient_flag = False

            env.world.tick()
            env.car_state = env.get_true_state(orient_flag)
            print(env.car_state[3])
            # env.cam_sensor.listen(lambda image: image.save_to_disk('/home/dhagash/MS-GE-02/MSR-Project/camera_pos_fix/%06d.png' % image.frame))
            env.nearest_wp_idx = env.calculate_nearest_index(env.car_state, waypoints)
            try:
                prev_states, prev_controls, prev_t, prev_v = env.mpc(orient_flag, prev_states, prev_controls, prev_t, prev_v, system_params, waypoints.shape[0] + 1, Ts)
                steer = prev_controls[1, 0] / 1.22

                if prev_controls[0, 0] > 0:
                    throttle = prev_controls[0, 0]
                    brake = 0
                else:
                    brake = prev_controls[0, 0]
                    throttle = 0

                env.vehicle.apply_control(carla.VehicleControl(throttle = throttle, steer = steer, reverse = False, brake = brake))

            except RuntimeError as err:
                print('Error occured with following error message: \n {} \n'.format(err))
                print('Fallback on Stanley Control !!!')
                steer = env.stanley_control(waypoints)
                if prev_controls[0, 0] > 0:
                    throttle = prev_controls[0, 0]
                    brake = 0
                else:
                    brake = prev_controls[0, 0]
                    throttle = 0

                env.vehicle.apply_control(carla.VehicleControl(
                throttle = throttle, steer = steer, reverse = False, brake = brake))

                prev_controls[0, :] = prev_controls[0, 0]
                prev_controls[1, :] = steer
                prev_states = np.vstack([env.car_state] * (env.P + 1)).T
                prev_t = prev_t + env.car_state[3] * Ts
                pass

            # env.states.append(state(0, env.car_state[0], env.car_state[1], env.car_state[2], env.car_state[3], env.car_state[4], 0))
            # env.controls.append(control(0, throttle, brake, env.control[0], steer, 0))

            if (env.nearest_wp_idx == waypoints.shape[0] - 1) and env.nearest_wp_idx != prev_idx:
                laps_completed += 1
                print('Lap {} completed.....'.format(laps_completed))
                prev_t = prev_t - prev_t[0]
                if laps_completed == num_of_laps:
                    print('Braking......')
                    while env.car_state[3] != 0.0:
                        env.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0, reverse = False, brake = 0.2))
                        env.car_state = env.get_true_state(orient_flag)
                        env.world.tick()
                    break
                print('starting lap {}......'.format(laps_completed + 1))
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