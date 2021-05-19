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
from scipy.sparse import csc_matrix

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

def nearest_state(curr_state, waypoints, vx_des, vy_des):
    l = len(waypoints)

    x_r = curr_state[0]
    y_r = curr_state[1]

    nearest_state = [0, 0, 0, MX(vx_des), MX(vy_des)]

    # Search nearest waypoint index
    dx = vcat([(x_r - pose[0])**2 for pose in waypoints])
    dy = vcat([(y_r - pose[1])**2 for pose in waypoints])
    dist_sq = dx + dy

    waypoints = MX(hcat(waypoints))
    
    minimum = inf
    idx_1 = 0
    for i in range(l):
        idx_1 = if_else(dist_sq[i] < minimum, i, idx_1)
        minimum = if_else(dist_sq[i] < minimum, dist_sq[i], minimum)

    idx_2 = if_else(dist_sq[idx_1 + 1] < dist_sq[idx_1 - 1], idx_1 + 1, idx_1 - 1)

    x1 = waypoints[0, idx_1]
    y1 = waypoints[1, idx_1]
    x2 = waypoints[0, idx_2]
    y2 = waypoints[1, idx_2]

    l = ((x2 ** 2) + (y2 ** 2) - (x1 * x2 + y1 * y2) + x_r * (x1 - x2) + y_r * (y1 - y2)) / (((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    
    nearest_state[0] = l * waypoints[0, idx_1] + (1 - l) * waypoints[0, idx_2]
    nearest_state[1] = l * waypoints[1, idx_1] + (1 - l) * waypoints[1, idx_2]
    nearest_state[2] = l * waypoints[2, idx_1] + (1 - l) * waypoints[2, idx_2]

    return MX(vcat(nearest_state))


class CarEnv():

    def __init__(self):
        """Initialize simulation environment
        """
        # Connect to client
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(2.0)

        # Get World
        self.world = self.client.get_world()

        # Get Map of the world
        self.map = self.world.get_map()
        
        # Get all the blueprints available in the world
        self.blueprint_library =  self.world.get_blueprint_library()
        
        # Initialize state dataclass list to log subsequent state information
        self.states = []

        # Initialize control dataclass list to log subsequent control commands
        self.controls = []

        # Data for longitudinal controller
        self.vel_control = [velocity_control_var(0.0, 0.0)]

        # Track errors
        self.errors = []


    def spawn_vehicle_2D(self, spawn_pose: np.ndarray):
        """Spawn a Vehicle at given 2D pose

        Arg:
            spawn_pose: Vehicle spawn pose [x, y, heading]
        """
        # Load Tesla Model 3 blueprint
        self.car_model = self.blueprint_library.find('vehicle.tesla.model3')

        # Spawn vehicle at given pose
        self.spawn_point = carla.Transform(carla.Location(spawn_pose[0], spawn_pose[1], 2), carla.Rotation(0, spawn_pose[2], 0))
        self.vehicle =  self.world.spawn_actor(self.car_model, self.spawn_point)

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
        self.data = np.loadtxt(path, delimiter=delimiter)
        return self.data


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
        for actor in self.actor_list:
            actor.destroy()


    # def nearest_state(self, x_r: float, y_r: float, yaw_r: float, waypoints: np.ndarray, vx_des, vy_des):
    #     """Compute the nearest pose to the vehicle on the desired trajectory

    #     Args:
    #         - x_r: Vehicle's x position in world frame
    #         - y_r: Vehicle's y position in world frame
    #         - yaw_r: Vehicle's yaw in world frame
    #         - waypoints: Desired trajectory coordinates [x, y, yaw] in world frame

    #     Returns:
    #         - nearest_pose: Nearest pose [x, y, yaw] on the desired trajectory to the vehicle
    #     """
    #     nearest_state = np.array([0, 0, 0, vx_des, vy_des])

    #     # Search nearest waypoint index
    #     dx = [x_r - x for x in waypoints[:, 0]]
    #     dy = [y_r - y for y in waypoints[:, 1]]
    #     dist_sq = np.power(dx, 2) + np.power(dy, 2)
    #     idx_1 = np.argmin(dist_sq)
    #     idx_2 = (idx_1 + 1) if (dist_sq[idx_1 + 1] < dist_sq[idx_1 - 1]) else (idx_1 - 1)

    #     x1, y1, yaw1 = waypoints[idx_1, :]
    #     x2, y2, yaw2 = waypoints[idx_2, :]

    #     l = ((x2 ** 2) + (y2 ** 2) - (x1 * x2 + y1 * y2) + x_r * (x1 - x2) + y_r * (y1 - y2)) / (((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    #     nearest_state[:3] = l * waypoints[idx_1, :] + (1 - l) * waypoints[idx_2, :]  

    #     return nearest_state.tolist()


    def get_true_state(self):
        """Get vehicles true state from the simulation
        
        Returns:
            - true_state: True state of the vehicle [x, y, yaw, vx, vy]
        """
        x = self.vehicle.get_transform().location.x
        y = self.vehicle.get_transform().location.y
        yaw = np.radians(self.vehicle.get_transform().rotation.yaw)     # [radians]
        vx = self.vehicle.get_velocity().x * np.cos(yaw) + self.vehicle.get_velocity().y * np.sin(yaw)
        vy = -self.vehicle.get_velocity().x * np.sin(yaw) + self.vehicle.get_velocity().y * np.cos(yaw)

        true_state = np.array([x, y, yaw, vx, vy])

        return true_state


    def mpc(self, waypoints: np.ndarray, laps_required: int):
        """Deploy Stanley Control Paradigm for vehicle control

        Args:
            - waypoints: Desired trajectory for vehicle (x, y, yaw)
            - laps_required: Number of laps to be completed
        """
        # Desired velocity [m/s]
        vx_des = 14.0
        vy_des = 0.0

        s = MX.sym('s', 5)
        x, y, yaw, vx, vy = s[0], s[1], s[2], s[3], s[4]

        u = MX.sym('u', 2)
        acc, delta = [u[0], u[1]]   


        prediction = vertcat(x + sqrt(vx**2 + vy**2) * cos(atan2(tan(delta), 2) + yaw) * 0.03,
                             y + sqrt(vx**2 + vy**2) * sin(atan2(tan(delta), 2) + yaw) * 0.03,
                             yaw + (sqrt(vx**2 + vy**2) * tan(delta) * 0.03 / sqrt((19.8025) + (4.95 * tan(delta)**2))),
                             vx + ((4.22 * acc) - (-0.0013 * sqrt(vx**2 + vy**2) * vx - 0.362 * vx)) * 0.03,
                             vy - (1.318 * (atan2(vy, vx) - delta) + (-0.0013 * sqrt(vx**2 + vy**2) + 0.362) * vy) * 0.03)

        pred = Function('pred', [s, u], [prediction])

        opti = Opti()

        N = 1
        s = opti.variable(5, N + 1)
        e = opti.variable(5, N + 1)
        u = opti.variable(2, N)
        p = opti.parameter(5, 1)

        opti.minimize(sumsqr(s))

        for k in range(N):
            opti.subject_to(e[:, k] == s[:, k] - nearest_state(s[:, k], waypoints, vx_des, vy_des))
            opti.subject_to(s[:, k+1] == pred(s[:, k], u[:, k]))

        opti.subject_to(s[:, 0] == p)

        # Set using true states
        opti.set_value(p, [waypoints[1][0],waypoints[1][1],waypoints[1][2], vx_des, vy_des])

        opti.subject_to( opti.bounded(0, u[0, :], 1) )
        opti.subject_to(opti.bounded(-1.22, u[1, :], 1.22))

        # Avoid NaN error
        opti.set_initial(s, 1)

        opti.solver('ipopt')

        sol = opti.solve()

        print(sol.value(s), sol.value(u))

        # Apply this
        control = u[:, 0]

        # Find true state
        # New parameter value p

        # Loop over

        # Wait for car to stabilize after spawning
        while self.vehicle.get_velocity().z != 0:
            pass
        

        # Desired trajectory
        x_des = waypoints[:,0]
        y_des = waypoints[:,1]
        yaw_des = [wrapToPi(np.radians(i)) for i in waypoints[:, 2]] # [radians] 
        
        laps_completed = 0

        # Prediction Horizon
        p = 5

        while 1:
            state = self.get_true_state()

            self.snapshot = self.world.wait_for_tick()
            curr_t = self.snapshot.timestamp.elapsed_seconds # [seconds]
            
            opti = Opti()
            U = opti.variable(p, 2)

            # Steering angle control
            _steer = np.degrees(wrapToPi(psi_h + psi_c))  # uncontrained in degrees
            _steer = max(min(_steer, self.max_steer_angle), -self.max_steer_angle)
            _steer = (_steer)/self.max_steer_angle # constrained to [-1, 1]

            # Split velocity control into throttle and brake and constrain them to [0, 1]
            if acc >= 0:
                _throttle = np.tanh(acc)
                _brake = 0
                if (_throttle - self.controls[-1].throttle) > 0.1:
                    _throttle = self.controls[-1].throttle + 0.1
            else:
                _throttle = 0
                _brake = np.tanh(abs(acc))
                if (_brake - self.controls[-1].brake) > 0.1:
                    _brake = self.controls[-1].brake + 0.1
            
            # Append control data 
            self.controls.append(control(curr_t, _throttle, _brake, _throttle - _brake, _steer * self.max_steer_angle, laps_completed))
            
            # Apply control
            self.vehicle.apply_control(carla.VehicleControl(
                throttle = _throttle, steer = _steer, reverse = False, brake = _brake))
            

def main():
    try:
        # Data logging flag
        save_flag = False

        # Initialize car environment
        env = CarEnv()

        # Load waypoints
        waypoints = env.read_file("../../Data/2D_waypoints.txt")

        # Spawn a vehicle at spawn_pose
        spawn_pose = waypoints[0]
        env.spawn_vehicle_2D(spawn_pose)

        # Set controller tuning params
        # Default params
        num_of_laps = 1
        kp, ki, kd, ke, kv = [0.5, 0.01, 0.15, 0.1, 5.0]

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
        curr_t = env.world.wait_for_tick().timestamp.elapsed_seconds # [seconds]
        env.states.append(state(curr_t, spawn_pose[0], spawn_pose[1], spawn_pose[2], 0.0, 0.0, 0))
        env.controls.append(control(curr_t, 0.0, 0.0, 0.0, 0.0, 0))

        # Initialize control loop
        env.mpc(waypoints, num_of_laps)
    
    finally:
        if save_flag:
        # Save all Dataclass object lists to a file
            env.save_log('../../Data/states_mpc.pickle', env.states)
            env.save_log('../../Data/controls_mpc.pickle', env.controls)

        # Destroy all actors in the simulation
        env.destroy()


if __name__=="__main__":
    main()