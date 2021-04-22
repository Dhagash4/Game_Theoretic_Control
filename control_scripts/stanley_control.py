# Author: Saurabh Gupta
# email: s7sagupt@uni-bonn.de
# MSR Project Sem 2

# Implementation of Stanley Vehicle trjactory tracking control loop

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
        self.d = 2.5                  # distance between rear and front wheel

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


def calculate_target_index(robot: Robot, xs_des: list, ys_des: list()):
    
    # Front axle position
    fy = robot.pos[1] + robot.d * np.sin(robot.theta)
    fx = robot.pos[0] + robot.d * np.cos(robot.theta)

    # search nearest point index to axle
    dx = [fx - x for x in xs_des]
    dy = [fy - y for y in ys_des]
    d = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    ind = np.argmin(d)

    # Project error between front axle and nearest point to the front axle vector
    front_axle_vec = [-np.cos(robot.theta + np.pi / 2), -np.sin(robot.theta + np.pi / 2)]
    error_front_axle = np.dot([dx[ind], dy[ind]], front_axle_vec)

    return ind, error_front_axle


def steer_controller(delta_des: float, delta: float, cumulative_error: float, prev_err: float, tuning_param: list, dt: float):
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

    # Compute error between current and desired value
    curr_err = delta_des - delta

    cumulative_error += curr_err

    # Steering rate control signal
    delta_dot = (kp * curr_err) + (kd * (curr_err - prev_err) / dt) + (ki * cumulative_error * dt)

    return delta_dot, curr_err, cumulative_error

if __name__=="__main__":
    # Robot initialization
    robot = Robot()
    robot.pos = np.array([90, 0])
    robot.theta = 0
    robot.v = 0
    robot.delta = 0

    # Desired velocity
    v_des = 20

    # Define desired trajectory
    angles = np.arange(0, 2 * pi, 0.05)
    x_des = [np.cos(angle) * 100 for angle in angles]
    y_des = [np.sin(angle) * 100 for angle in angles]
    yaw_des = [np.arctan2(x_, -y_) for (x_, y_) in zip(x_des, y_des)]

    # Variables to log robot states
    x_traj = [robot.pos[0]]
    y_traj = [robot.pos[1]]
    theta_traj = [robot.theta]
    v_traj = [robot.v]
    delta_traj = [robot.delta]

    # Control Time Step
    T = 100
    dt = 0.1 # 10 Hz
    t = 0

    err_v = v_des - robot.v
    err_total_v = 0

    err_delta = 0
    err_total_delta = 0

    # Crosstrack error control gain
    # Smooth recovery
    k = 0.8

    # PID tuning parameters
    kp_lon, kd_lon, ki_lon = 4.0, 0.15, 0.5
    kp_lat, kd_lat, ki_lat = 2.5, 0.0, 0.0

    # Control Command History
    acc_hist = [0]
    steer_hist = [0]

    last_idx = 0
    while t < T and last_idx < len(x_des) - 1:
        
        # TODO: compute control and update robot pose
        acc, err_v, err_total_v = longitudinal_controller(robot.v, v_des, err_v, err_total_v, [kp_lon, ki_lon, kd_lon], dt)
        
        idx, error_front_axle = calculate_target_index(robot, x_des, y_des)

        if(idx <= last_idx):
            idx = last_idx
        else:
            last_idx = idx

        # Heading error
        psi_h = wrapToPi(yaw_des[idx] - robot.theta)

        # Crosstrack error
        psi_c = np.arctan2(k * error_front_axle, 0.5 + robot.v)

        delta_des = psi_h + psi_c

        # print(idx, yaw_des[idx], robot.theta, psi_h, psi_c, delta_des)
        
        delta_dot, err_delta, err_total_delta = steer_controller(delta_des, robot.delta, err_total_delta, err_delta, [kp_lat, ki_lat, kd_lat], dt)

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
    np.savetxt('data_stanley.txt', data[0])
    
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