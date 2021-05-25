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

def save_log(filename: str, data: object):
        """Logging data to a .pickle file

        Args:
            - filename: Name of the file to store data
            - data: Data to be logged
        """
        with open(filename, "wb") as f:
            pickle.dump(data, f)

def predict_new(old_states, control, params, dt):
    L, p, Cd, Cfr, Ccs = params
    new_states = np.zeros_like(old_states)

    x, y, theta, vx, vy = old_states
    v = np.sqrt(vx ** 2 + vy ** 2)
    acc, delta = control
    x_new = x + (v * np.cos(np.arctan2(np.tan(delta), 2) + theta) * dt)
    y_new = y + (v * np.sin(np.arctan2(np.tan(delta), 2) + theta) * dt)
    theta_new = wrapToPi(theta + (v * np.tan(delta) * dt / np.sqrt((L ** 2) + ((0.5 * L * np.tan(delta)) ** 2))))
    vx_new = vx + (p * acc - Cd * v * vx - Cfr * vx) * dt
    vy_new = vy - (Ccs * wrapToPi(np.arctan2(vy, vx) - delta) + (Cd * v + Cfr) * vy) * dt
    new_states = np.array([x_new, y_new, theta_new, vx_new, vy_new])

    return new_states


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


if __name__ == '__main__':
    waypoints = np.loadtxt('../../Data/2D_waypoints.txt')
    
    states = []
    controls = []

    params = np.array([4, 10, 0, 0, 0])

    spawn_pose = waypoints[10, :]

    states.append(state(0, spawn_pose[0], spawn_pose[1], spawn_pose[2], 0, 0, 0))
    controls.append(control(0, 0.0, 0.0, 0.0, 0.0, 0))

    curr_idx = 10
    curr_state = np.array([spawn_pose[0], spawn_pose[1], spawn_pose[2], 0.01, 0.01])

    # Desired velocity [m/s]
    vx_des = 10.0
    vy_des = 0.0

    s = MX.sym('s', 5)
    x, y, yaw, vx, vy = s[0], s[1], s[2], s[3], s[4]

    u = MX.sym('u', 2)
    acc, delta = u[0], u[1]

    prediction = vertcat(x + sqrt(vx**2 + vy**2) * cos(atan2(tan(delta), 2) + yaw) * 0.03,
                            y + sqrt(vx**2 + vy**2) * sin(atan2(tan(delta), 2) + yaw) * 0.03,
                            atan2(sin(yaw + (sqrt(vx**2 + vy**2) * tan(delta) * 0.03 / sqrt((19.8025) + (4.95 * tan(delta)**2)))), cos(yaw + (sqrt(vx**2 + vy**2) * tan(delta) * 0.03 / sqrt((19.8025) + (4.95 * tan(delta)**2))))),
                            vx + ((4.22 * acc) - (-0.0013 * sqrt(vx**2 + vy**2) * vx - 0.362 * vx)) * 0.03,
                            vy - (1.318 * (atan2(vy, vx) - delta) + (-0.0013 * sqrt(vx**2 + vy**2) + 0.362) * vy) * 0.03)
    

    pred = Function('pred', [s, u], [prediction])

    opti = Opti()

    # Prediction Horizon
    i = 0
    while(i < 300):
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
            opti.subject_to(s[:, k+1] == pred(s[:, k], u[:, k]))

        opti.subject_to(s[:, 0] == p)

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

        # self.snapshot = self.world.wait_for_tick()
        # curr_t = self.snapshot.timestamp.elapsed_seconds # [seconds]
        
        # Append control data 
        controls.append(control(0, throttle, brake, throttle - brake, steer * 1.22, 0))
        
        print('Apply control')
        curr_state = predict_new(curr_state, cont[:, 0], params, 0.03)
        states.append(state(0, curr_state[0], curr_state[1], curr_state[2], curr_state[3], curr_state[4], 0))

    save_log('states.pickle', states)
    save_log('controls.pickle', controls)