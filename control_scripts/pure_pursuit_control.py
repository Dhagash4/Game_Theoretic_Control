# Author: Saurabh Gupta
# email: s7sagupt@uni-bonn.de
# MSR Project Sem 2

# Implementation of Pure Pursuit trajectory tracking control loop

import numpy as np
from math import fabs, pi, hypot
from matplotlib import pyplot as plt

from util import wrapToPi, euclidean_distance

# Bicycle model
class Robot:
    def __init__(self):
        self.pos = np.zeros(2)      # x, y position of the robot
        self.theta = 0              # yaw angle of the robot [rad]
        self.v = 0                  # velocity of the robot
        self.delta = 0              # steering angle of front wheels [rad]
        self.delta_max = pi / 6     # max steering angle
        self.d = 1                  # distance between rear and front wheel

    def apply_control(self, a: float, steer_rate: float, dt: float):
        # Update velocity and steering angle
        self.v = self.v + a * dt
        self.delta = self.delta + steer_rate * dt

        # Steering constraint
        if fabs(self.delta) > self.delta_max:
            self.delta = np.sign(self.delta) * self.delta_max

        # Update pose
        self.pos = self.pos + (self.v * np.array([np.cos(self.theta), np.sin(self.theta)]) * dt)
        self.theta = wrapToPi(self.theta + ((self.v * np.tan(self.delta) / self.d) * dt))


# Linear Velocity Control
def longitudinal_controller(v: float, v_des: float, prev_err: float, cumulative_error: float, tuning_param: list, dt: float):
    """
    Compute control signal (acceleratio/deceleration) for linear velocity control
    Args:
        - v (float): current velocity
        - v_des (float): velocity setpoint
        - prev_error (float): velocity error from previous control loop iteration for derivative controller
        - cumulative_error (float): accumulated error over all previous control loop iterations for integral controller
        - tuning_param (1x3 array of floats): [kp, ki, kd]: PID controller tuning parameters
        - dt (float): Controller time step
    Returns:
        - acc (float): acceleration/deceleration control signal
        - curr_err (float): velocity error from the current control loop
        - cumulative_error (float): accumulated error upto current control loop
    """
    # Extract PID tuning parameters
    [kp, ki, kd] = tuning_param

    # Compute error between current and desired value
    curr_err = v_des - v

    cumulative_error += curr_err

    # Acceleration/Deceleration control signal
    acc = (kp * curr_err) + (kd * (curr_err - prev_err) / dt) + (ki * cumulative_error * dt)

    return acc, curr_err, cumulative_error


def calculate_lookahead_index(robot: Robot, xs_des: list, ys_des: list(), Ld: float):
    # search nearest point index
    dx = [robot.pos[0] - icx for icx in xs_des]
    dy = [robot.pos[1] - icy for icy in ys_des]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))
    distance_this_index = euclidean_distance(robot.pos, np.array([xs_des[ind], ys_des[ind]]))
    while True:
        ind = ind + 1 if (ind + 1) < len(xs_des) else ind
        distance_next_index = euclidean_distance(robot.pos, np.array([xs_des[ind], ys_des[ind]]))
        if distance_this_index <= distance_next_index:
            break
        distance_this_index = distance_next_index

    # search look ahead target point index
    L = 0
    while Ld > L and (ind + 1) < len(xs_des):
        dx = xs_des[ind] - robot.pos[0]
        dy = ys_des[ind] - robot.pos[1]
        L = hypot(dx, dy)
        ind += 1

    return ind


def lateral_controller(l_d: float, alpha: float, delta: float, cumulative_error: float, prev_err: float, tuning_param: list, dt: float):
    """
    Compute control signal (steering rate) for lateral position control
    Args:
        - l_d (float): distance from robot to lookahead point on desired trajectory
        - alpha (float):  angle between robot heading and the direction from robot to lookahead point on desired trajectory [rad]
        - delta (float): robot's current steering angle [rad]
        - prev_error (float): steering angle error from previous control loop iteration for derivative controller
        - cumulative_error (float): accumulated error over all previous control loop iterations for integral controller
        - tuning_param (1x3 array of floats): [kp, ki, kd]: PID controller tuning parameters
        - dt (float): Controller time step
    Returns:
        - delta_dot (float): steering rate control signal [rad/sec]
        - curr_err (float): steering angle error from the current control loop
        - cumulative_error (float): accumulated error upto current control loop
    """ 
    # Extract PID tuning parameters
    [kp, ki, kd] = tuning_param

    # Compute the desired steering angle for the lookahead point on desired trajectory
    delta_des = np.arctan2(2 * robot.d * np.sin(alpha) , l_d)

    # Compute error between current and desired value
    curr_err = delta_des - delta

    cumulative_error += curr_err

    # Steering rate control signal
    delta_dot = (kp * curr_err) + (kd * (curr_err - prev_err) / dt) + (ki * cumulative_error * dt)

    return delta_dot, curr_err, cumulative_error

if __name__=="__main__":
    # Robot initialization
    robot = Robot()
    robot.pos = np.array([100, 0])
    robot.theta = pi / 2
    robot.v = 0
    robot.delta = 0

    # Desired velocity
    v_des = 10

    # Define desired trajectory
    angles = np.arange(0, 2 * pi, 0.05)
    x_des = [np.cos(angle) * 100 for angle in angles]
    y_des = [np.sin(angle) * 100 for angle in angles]

    # Variables to log robot states
    x_traj = [robot.pos[0]]
    y_traj = [robot.pos[1]]
    theta_traj = [robot.theta]
    v_traj = [robot.v]
    delta_traj = [robot.delta]

    # Control Time Step
    T = 75
    dt = 0.1 # 10 Hz
    t = 0

    err_v = v_des - robot.v
    err_total_v = 0

    err_delta = 0
    err_total_delta = 0

    Ld = 2

    kp_lon, kd_lon, ki_lon = 4.0, 0.15, 0.5
    kp_lat, kd_lat, ki_lat = 1.5, 0.01, 0.01

    # Control Command History
    acc_hist = [0]
    steer_hist = [0]

    while t < T:
        # TODO: compute control and update robot pose
        acc, err_v, err_total_v = longitudinal_controller(robot.v, v_des, err_v, err_total_v, [kp_lon, ki_lon, kd_lon], dt)
        
        lookahead_idx = calculate_lookahead_index(robot, x_des, y_des, Ld)
        l_d = euclidean_distance(robot.pos, np.array([x_des[lookahead_idx], y_des[lookahead_idx]]))
        alpha = wrapToPi(np.arctan2((y_des[lookahead_idx] - robot.pos[1]), (x_des[lookahead_idx] - robot.pos[0])) - robot.theta)
        
        delta_dot, err_delta, err_total_delta = lateral_controller(l_d, alpha, robot.delta, err_total_delta, err_delta, [kp_lat, ki_lat, kd_lat], dt)
            
        robot.apply_control(acc, delta_dot, dt)

        # update time
        t = t + dt
        
        # save varaibles for plotting
        x_traj.append(robot.pos[0])
        y_traj.append(robot.pos[1])
        theta_traj.append(robot.theta)
        v_traj.append(robot.v)
        delta_traj.append(robot.delta)
        
        acc_hist.append(acc)
        steer_hist.append(delta_dot)

    data = np.dstack((x_traj, y_traj, theta_traj, v_traj, delta_traj, acc_hist, steer_hist))
    print(data)
    np.savetxt('data.txt', data[0])
    
    # Plot trajectory
    plt.cla()
    # ex.plot_arrow(robot.x, robot.y, robot.theta)
    plt.plot(x_des, y_des, ".r", label="course")
    plt.plot(x_traj, y_traj, "-b", label="trajectory")
    plt.legend()
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()