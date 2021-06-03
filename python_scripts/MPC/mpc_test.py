import glob
import getopt
import pickle
import os, sys
import numpy as np
from casadi import *
from numpy.lib.utils import info
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt
sys.path.append('..')

from Common.util import *
from Common.custom_dataclass import *

class CarEnv():
    def __init__(self):
        """Initialize simulation environment
        """
        
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
        # Spawn vehicle at given pose
        self.spawn_pose = self.waypoints[self.spawn_idx]
        self.car_pose = np.r_[self.spawn_pose, 0.0001, 0.0001]

    def save_log(self, filename: str, data: object):
            """Logging data to a .pickle file

            Args:
                - filename: Name of the file to store data
                - data: Data to be logged
            """
            with open(filename, "wb") as f:
                pickle.dump(data, f)

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

    def predict_new(self, old_states, control, params, dt):
        L, p, Cd, Cfr, Ccs = params

        x, y, theta, vx, vy = old_states
        v = np.sqrt(vx ** 2 + vy ** 2)
        acc, delta = control

        x_new = x + (v * np.cos(np.arctan2(np.tan(delta), 2) + theta) * dt)
        y_new = y + (v * np.sin(np.arctan2(np.tan(delta), 2) + theta) * dt)
        theta_new = wrapToPi(theta + (v * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
        vx_new = vx + (p * acc - Cd * v * vx - Cfr * vx) * dt
        vy_new = vy - (Ccs * wrapToPi(np.arctan2(vy, vx) - delta) + (Cd * v + Cfr) * vy) * dt
        self.car_pose = np.array([x_new, y_new, theta_new, vx_new, vy_new])


    def calculate_error(self, state_mpc, theta):
        x = state_mpc[0]
        y = state_mpc[1]

        x_ref = self.lut_x(theta)
        y_ref = self.lut_y(theta)
        yaw = self.lut_theta(theta)

        e_c = sin(yaw) * (x - x_ref) - cos(yaw) * (y - y_ref)
        e_l = -cos(yaw) * (x - x_ref) - sin(yaw) * (y - y_ref)

        return MX(vcat([e_c**2, e_l**2]))

    def calculate_nearest_index(self):
        # Search nearest waypoint index
        x_r, y_r = self.car_pose[:2]
        # fromIdx = max(0, self.nearest_wp_idx - 10)
        # toIdx = self.nearest_wp_idx + 10

        dx = np.array([(x_r - x) ** 2 for x in self.waypoints[:, 0]])
        dy = np.array([(y_r - y) ** 2 for y in self.waypoints[:, 1]])
        dist = dx + dy

        self.nearest_wp_idx = np.argmin(dist)# + fromIdx)# % self.waypoints.shape[0]

    def set_mpc_params(self, P, C, vmax):
        self.P = P
        self.C = C
        self.vmax = vmax
        self.theta = 0
        
    def set_opti_weights(self, w_u0, w_u1, w_qc, w_ql, gamma, w_c):
        self.w_u0 = w_u0
        self.w_u1 = w_u1
        self.w_qc = w_qc
        self.w_ql = w_ql
        self.gamma = gamma
        self.w_c = w_c

    def fit_curve(self):
        self.L = np.arange(0, self.waypoints.shape[0])

        self.lut_x = interpolant('LUT_x', 'bspline', [self.L], self.waypoints[:, 0], dict(degree=[3]))
        self.lut_y = interpolant('LUT_y', 'bspline', [self.L], self.waypoints[:, 1], dict(degree=[3]))
        self.lut_theta = interpolant('LUT_t', 'bspline', [self.L], self.waypoints[:, 2], dict(degree=[1]))

    def mpc(self, Ts):
        """Deploy Stanley Control Paradigm for vehicle control

        Args:
            - waypoints: Desired trajectory for vehicle (x, y, yaw)
            - laps_required: Number of laps to be completed
        """
        err = [0, 0]
        pose = []
        
        s = MX.sym('s', 5)
        x, y, yaw, vx, vy = s[0], s[1], s[2], s[3], s[4]

        u = MX.sym('u', 2)
        acc, delta = [u[0], u[1]]

        dt = MX.sym('dt')

        prediction = vertcat(x + sqrt((vx + 0.00001) ** 2 + (vy + 0.00001)** 2) * cos(atan2(tan(delta), 2) + yaw) * dt,
                            y + sqrt((vx + 0.00001) ** 2 + (vy + 0.00001) ** 2) * sin(atan2(tan(delta), 2) + yaw) * dt,
                            yaw + (sqrt((vx + 0.00001) ** 2 + (vy + 0.00001) ** 2) * tan(delta) * dt / sqrt((19.8025) + (4.95 * tan(delta) ** 2))),
                            vx + ((4.22 * acc) - (-0.0013 * sqrt((vx + 0.00001) ** 2 + (vy + 0.00001) ** 2) * vx - 0.362 * vx)) * dt,
                            vy - (1.318 * (atan2(vy, vx+0.00001) - delta) + (-0.0013 * sqrt((vx + 0.00001) ** 2 + (vy + 0.00001) ** 2) + 0.362) * vy) * dt)
    
        pred = Function('pred', [s, u, dt], [prediction])

        opti = Opti()

        s = opti.variable(5, self.P + 1)
        e = opti.variable(2, 1)
        # e = opti.variable(2, self.P + 1)
        t = opti.variable(1, self.P + 1)
        v = opti.variable(1, self.P + 1)
        u = opti.variable(2, self.C)
        p = opti.parameter(5, 1)

        opti.minimize(self.w_qc * e[0] + self.w_ql * e[1]
                     #+ self.w_qc * sumsqr(e[0, :]) + self.w_ql * sumsqr(e[1, :])
                     - self.gamma * sum2(v) * Ts
                    #  + self.w_u0 * sumsqr(u[0, :]) + self.w_u1 * sumsqr(u[1, :])
                     + self.w_c * sumsqr(u[:, :self.C-1] - u[:, 1:self.C])
                     + self.w_c * sumsqr(v[:self.P] - v[1:self.P + 1])
                     )
        
        # Set using true states
        # pose.append(self.car_pose)
        opti.set_value(p, self.car_pose)
        opti.subject_to(s[:, 0] == p)
        opti.subject_to(t[0] == self.theta)

        # print('read true state')
        opti.subject_to(opti.bounded(-1.0, u[0, :], 1.0))
        opti.subject_to(opti.bounded(-1.22, u[1, :], 1.22))
        opti.subject_to(opti.bounded(0, t, self.waypoints.shape[0] + 1))
        opti.subject_to(v <= self.vmax)
        opti.subject_to(v >= 0)

        for i in range(self.C):
            if i < 0.6 * self.P:
                self.dt = 0.1
            else:
                self.dt = 0.1

            # err += self.calculate_error(pose[-1], t[i])
            # pose.append(pred(pose[-1], u[:, i], self.dt))
            opti.subject_to(s[:, i+1] == pred(s[:, i], u[:, i], self.dt))
            err += self.calculate_error(s[:, i], t[i])
            # opti.subject_to(e[:, i] == self.calculate_error(s[:, i], t[i]))
            opti.subject_to(t[i + 1] == t[i] + v[i] * Ts)

        for i in range(self.C, self.P):
            if i < 0.6 * self.P:
                self.dt = 0.1
            else:
                self.dt = 0.1

            # err += self.calculate_error(pose[-1], t[i])
            # pose.append(pred(pose[-1], u[:, self.C - 1], self.dt))
            opti.subject_to(s[:, i+1] == pred(s[:, i], u[:, self.C - 1], self.dt))
            err += self.calculate_error(s[:, i], t[i])
            # opti.subject_to(e[:, i] == self.calculate_error(s[:, i], t[i]))
            opti.subject_to(t[i + 1] == t[i] + v[i] * Ts)

        # opti.subject_to(e[:, -1] == self.calculate_error(s[:, -1], t[-1]))
        # err += self.calculate_error(pose[-1], t[-1])
        err += self.calculate_error(s[:, -1], t[-1])
        opti.subject_to(e == err)

        # Good Initialization
        opti.set_initial(s, np.vstack([self.car_pose] * (self.P + 1)).T)
        opti.set_initial(u, np.vstack([self.control] * self.C).T)
        opti.set_initial(t, self.nearest_wp_idx)
        opti.set_initial(v, self.car_pose[3])

        p_opts = {"print_time": False, 'ipopt.print_level': 0}
        opti.solver('ipopt', p_opts)

        sol = opti.solve()

        # print("solution found")
        self.control = sol.value(u)[:, 0]
        self.theta = sol.value(t)[1]
        # self.theta = self.nearest_wp_idx

        plt.plot(sol.value(s)[0, :], sol.value(s)[1, :])
        plt.pause(0.1)
        # print('states: \n', sol.value(s))
        print('errors: \n', sol.value(e))
        print('controls: \n', sol.value(u)[:, 0])
        print('path progress: \n', sol.value(t))
        print('velocity: \n', sol.value(v))


if __name__ == '__main__':
    # Data logging flag
    save_flag = False

    # Initialize car environment
    env = CarEnv()

    # Load waypoints
    env.waypoints = env.read_file("../../Data/2D_waypoints.txt")
    
    env.spawn_idx = 0
    env.spawn_vehicle_2D()

    params = np.loadtxt('../../Data/params.txt')

    # Spawn a vehicle at spawn_pose
    env.nearest_wp_idx = env.spawn_idx

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
    
    env.states.append(state(0, env.spawn_pose[0], env.spawn_pose[1], env.spawn_pose[2], 0.0, 0.0, 0))
    env.controls.append(control(0, 0.0, 0.0, 0.0, 0.0, 0))
    
    # Initialize control loop
    env.set_mpc_params(P = 15, C = 15, vmax = 35)
    env.set_opti_weights(w_u0 = 1, w_u1 = 1, w_qc = 1, w_ql = 2, gamma = 5, w_c = 2)

    env.fit_curve()
    env.dt = 0.1

    i = 0
    try:
        while(i < 500):
            i += 1
            env.calculate_nearest_index()
            env.mpc(env.dt)

            if env.control[0] > 0:
                throttle = env.control[0]
                brake = 0
            else:
                brake = env.control[0]
                throttle = 0

            steer = env.control[1] / 1.22

            env.states.append(state(0, env.car_pose[0], env.car_pose[1], env.car_pose[2], env.car_pose[3], env.car_pose[4], 0))
            env.controls.append(control(0, throttle, brake, env.control[0], steer, 0))

            env.predict_new(env.car_pose, env.control, params, env.dt)

            # if (env.nearest_wp_idx == env.waypoints.shape[0] - 1) and env.nearest_wp_idx != prev_idx:
            #     laps_completed += 1
            #     if laps_completed == num_of_laps:
            #         while env.car_pose[3] != 0.0:
            #             env.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0, reverse = False, brake = 0.2))
            #             env.get_true_state()
            #             env.world.tick()
            #         break

            # prev_idx = env.nearest_wp_idx

    finally:
        if save_flag:
        # Save all Dataclass object lists to a file
            env.save_log('../../Data/states_mpc.pickle', env.states)
            env.save_log('../../Data/controls_mpc.pickle', env.controls)

            print('saving...')