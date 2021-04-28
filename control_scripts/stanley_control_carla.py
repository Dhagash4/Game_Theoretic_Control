import glob
import os
import sys
import random
import time
import numpy as np
from util import *
from datetime import datetime

#Import CARLA anywhere on the system
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarEnv():

    def __init__(self, waypoints):
        ''' Initialize car environment
        '''
        # Connect to client
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(2.0)

        # Get World
        self.world = self.client.get_world()

        # Get Map of the world
        self.map = self.world.get_map()
        
        # Get all the blueprints available in the world
        blueprint_library =  self.world.get_blueprint_library()

        # Load Tesla Model 3 blueprint
        self.car_model = blueprint_library.find('vehicle.tesla.model3')

        # Spawn vehicle at first waypoint
        self.spawn_point = carla.Transform(carla.Location(waypoints[0, 0], waypoints[0, 1], 2), carla.Rotation(0, waypoints[0, 2], 0))
        self.vehicle =  self.world.spawn_actor(self.car_model, self.spawn_point)

        # Get max steering angle of car's front wheels
        self.max_steer_angle = self.vehicle.get_physics_control().wheels[0].max_steer_angle
        
        # List of actors which contains sensors and vehicles in the environment
        self.actor_list = []
        # Append our vehicle to actor list
        self.actor_list.append(self.vehicle)

        # Previous states
        self.prev_t = 0.0
        self.prev_v_error = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.int_error_v = 0.0
    
    def destroy(self):
        ''' Destroy all actors in the world
        '''
        for actor in self.actor_list:
            actor.destroy()


    def longitudinal_controller(self,v: float, v_des: float, prev_err: float, cumulative_error: float, tuning_param: list, dt: float):
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

        cumulative_error += (curr_err * dt)

        # Acceleration/Deceleration control signal
        acc = (kp * curr_err) + (kd * (curr_err - prev_err) / dt) + (ki * cumulative_error)

        return acc, curr_err, cumulative_error


    def calculate_target_index(self, x_r, y_r, xs_des: list, ys_des: list(), lookahead = 2):
        # Search nearest waypoint index
        dx = [x_r - x for x in xs_des]
        dy = [y_r - y for y in ys_des]
        dist = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        min_ind = (np.argmin(dist) + lookahead) % len(xs_des)
        min_dist = dist[min_ind]

        return min_ind, min_dist

    # def logging_data(self,idx,d,psi_h,yaw_diff,yaw_v,yaw_des,psi_c,_throttle,_brake,ke,x_des,y_des,x_v,y_v,_steer):
    #         with open("datalog_run_%f.txt"%(ke),"a+") as f:
    #             # f.write("ID:%d, Distance:%f, Heading_Error:%f, Yaw_Difference:%f,Yaw_Vehicle:%f, Yaw_des:%f, Cross_Track_Error:%f, Steer:%f, Throttle:%f, Brake:%f,Vehicle_x:%f,Vehicle_y:%f,x_des:%f,y_des:%f\n"%(idx,d,psi_h,yaw_diff,yaw,yaw_des,psi_c,_steer,_throttle,_brake))


    #             f.write("ID:%d, Distance:%f, Vehicle_x:%f, Vehicle_y:%f, Yaw_Vehicle:%f, x_des:%f, y_des:%f, Yaw_des:%f, Heading_Error:%f, Yaw_Difference:%f,  Cross_Track_Error:%f, Steer:%f, Throttle:%f, Brake:%f\n"%(idx, d, x_v, y_v, yaw_v, x_des, y_des, yaw_des, psi_h, yaw_diff, psi_c, _steer, _throttle, _brake))

    #             f.close()

    def stanley_control(self, waypoints):
        #Applying Control to the Car

        # Desired velocity [m/s]
        v_des = 14.0

        # Desired trajectory
        x_des = waypoints[:,0]
        y_des = waypoints[:,1]
        yaw_des = [wrapToPi(np.radians(i)) for i in waypoints[:, 2]] # [radians] 
        
        # Initial velocity error
        err_v = v_des - np.sqrt((self.vehicle.get_velocity().x) ** 2 + (self.vehicle.get_velocity().y) ** 2)

        err_total_v = 0

        # Crosstrack error control gains
        # Smooth recovery
        self.ke = 0.1
        # Low velocity gain
        self.kv = 10

        # PID tuning parameters
        kp_lon, kd_lon, ki_lon = 0.5, 0.1, 0.0

        self.track_idx = []
        self.track_d = []
        self.track_y_v = []
        self.track_x_v = []
        self.track_yaw_v = []
        self.track_v = []
                
        self.track_x_des = []        
        self.track_y_des = []
        self.track_yaw_des = []

        self.track_steer = []
        self.track_throttle = []
        self.track_brake = []
        self.track_head_err = []
        self.track_cross_err = []

        i = 0
        while 1:
            i += 1
            x = self.vehicle.get_transform().location.x
            y = self.vehicle.get_transform().location.y
            yaw = self.vehicle.get_transform().rotation.yaw     # [degrees]

            self.track_x_v.append(x)
            self.track_y_v.append(y)
            self.track_yaw_v.append(yaw)
            
            v = np.sqrt((self.vehicle.get_velocity().x) ** 2 + (self.vehicle.get_velocity().y) ** 2)
            self.track_v.append(v)

            self.snapshot = self.world.wait_for_tick()
            curr_t = self.snapshot.timestamp.elapsed_seconds # [seconds]
            # print(curr_t,self.prev_t)
            # Velocity control
            dt = curr_t - self.prev_t
            # dt = 1.0            
            acc, self.prev_v_error, self.int_error_v = self.longitudinal_controller(
                v, v_des, self.prev_v_error, self.int_error_v, [kp_lon, ki_lon, kd_lon], dt)

            # Find nearest waypoint
            idx, d = self.calculate_target_index(x, y, x_des, y_des, 3)

            self.track_idx.append(idx)
            self.track_d.append(d)
            self.track_x_des.append(x_des[idx])
            self.track_y_des.append(y_des[idx])
            self.track_yaw_des.append(np.degrees(yaw_des[idx]))

            self.world.debug.draw_string(carla.Location(waypoints[idx, 0], waypoints[idx, 1], 2), '.', draw_shadow=False,
                                   color=carla.Color(r=255, g=0, b=0), life_time=5,
                                   persistent_lines=True)
            # Heading error [radians]
            psi_h = wrapToPi(yaw_des[idx] - np.radians(yaw))

            self.track_head_err.append(psi_h)

            # Crosstrack yaw difference to path yaw [radians]
            yaw_diff = wrapToPi(yaw_des[idx] - np.arctan2(y - y_des[idx], x - x_des[idx]))
        
            # Crosstrack error in yaw [radians]
            psi_c = np.arctan2(self.ke * np.sign(yaw_diff) * d, self.kv + v)

            self.track_cross_err.append(psi_c)

            # Steering angle control
            _steer = np.degrees(wrapToPi(psi_h + psi_c))  # uncontrained in degrees
            _steer = max(min(_steer, self.max_steer_angle), -self.max_steer_angle)
            _steer = (_steer)/self.max_steer_angle # constrained to [-1, 1]
            self.track_steer.append(_steer)

            # Split velocity control into throttle and brake and constrain them to [0, 1]
            if acc >= 0:
                _throttle = np.tanh(acc)
                _brake = 0
            else:
                _throttle = 0
                _brake = np.tanh(abs(acc))
            
            if(_brake == 0):
                if (_throttle - self.prev_throttle) > 0.1:
                    _throttle = self.prev_throttle + 0.1
            
            if(_throttle == 0):
                if (_brake - self.prev_brake) > 0.1:
                    _brake = self.prev_brake + 0.1
            
            self.prev_throttle = _throttle
            self.prev_brake = _brake
            self.track_throttle.append(_throttle)
            self.track_brake.append(_brake)

            # self.logging_data(idx,d,psi_h,yaw_diff,psi_c,_throttle,_brake,yaw,yaw_des[idx],ke,x_des[idx],y_des[idx],x,y,_steer)
            
            # Apply control
            self.vehicle.apply_control(carla.VehicleControl(
                throttle = _throttle, steer = _steer, reverse = False, brake = _brake))

            self.prev_t = curr_t
            
        #After Control Information

        
def read_waypoints_file(path):
    ''' Read waypoints list
    '''
    waypoints = np.loadtxt(path)
    return waypoints

        
def main():
    try:
        # Load waypoints
        waypoints = read_waypoints_file("2D_waypoints.txt")
        # Initialize car environment
        env = CarEnv(waypoints)

        env.stanley_control(waypoints)
    
    finally:
        data_log_vehicle = np.vstack((env.track_x_v, env.track_y_v, env.track_yaw_v, env.track_v))
        data_log_waypoint = np.vstack((env.track_idx, env.track_d, env.track_x_des, env.track_y_des, env.track_yaw_des))
        data_log_error = np.vstack((env.track_head_err, env.track_cross_err))
        data_log_control = np.vstack((env.track_steer, env.track_throttle, env.track_brake))
        
        np.savetxt("datalog_veh_%f_%f.txt"%(env.ke, env.kv), data_log_vehicle.T)
        np.savetxt("datalog_way_%f_%f.txt"%(env.ke, env.kv), data_log_waypoint.T)
        np.savetxt("datalog_err_%f_%f.txt"%(env.ke, env.kv), data_log_error.T)
        np.savetxt("datalog_con_%f_%f.txt"%(env.ke, env.kv), data_log_control.T)
        env.destroy()

if __name__=="__main__":
    main()