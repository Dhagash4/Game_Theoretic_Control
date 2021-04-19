import glob
import os
import sys
import random
import time
import cv2
import numpy as np
import pygame

#Import CARLA anywhere on the system
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla




class CarEnv():


    def __init__(self):
        
        self.pressed_up = False
        self.pressed_down = False
        self.pressed_left = False
        self.pressed_right = False
        self.VMAX = 1.0
        self.WMAX = 0.7
        self.v = 0
        self.w = 0
        self.a=10
        self.al=100
        self.thres_v = 0.05
        self.thres_w = 0.05
        self.dt = 0.002 #secs
        
        
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(2.0)

        self.reduce_v= 0
        self.reduce_w = 0
        #World we want to have

        self.world = self.client.get_world()

        #Selecting the vehicle

        blueprint_library =  self.world.get_blueprint_library()
        self.car_model = blueprint_library.filter('model3')[0]
    
    # def car(self):    

        #list of actors which contains sensors and vehicles in the environment

        self.actor_list = []

        # Spawning car at random location in the map

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle =  self.world.spawn_actor(self.car_model,self.transform)
        self.actor_list.append(self.vehicle)


        #RGB Sensor 

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x','640')
        self.rgb_cam.set_attribute('image_size_y', '480')
        self.rgb_cam.set_attribute('fov', '110')
        

        #Attaching sensor to car

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.cam_sensor)

        # Data from the camera

        self.cam_sensor.listen(lambda image: image.save_to_disk('/home/dhagash/MS-GE-02/MSR-Project/output/%06d.png' % image.frame))

        #IMU Sensor

        self.imu = self.world.get_blueprint_library().find('sensor.other.imu')


        #Attaching imu to car

        self.imu_sensor = self.world.spawn_actor(self.imu, transform, attach_to=self.vehicle)
        self.actor_list.append(self.imu_sensor)


        #GNSS Sensor


        self.gnss = self.world.get_blueprint_library().find('sensor.other.gnss')



        #Attaching gnss to car

        self.gnss_sensor = self.world.spawn_actor(self.gnss, transform, attach_to=self.vehicle)
        self.actor_list.append(self.gnss_sensor)

        # time.sleep(15)

    # def process_img(self, image):
    #     i = np.array(image.raw_data)
    #     i2 =  i.reshape((480,640,4))
    #     i3 = i2[:,:,:3]

    #     cv2.imshow("",i3)
    #     cv2.waitKey(1)

    #     return i3/255.0

        # print("Collecting Image")

    def destroy(self):

        for actors in self.actor_list:
            actors.destroy()

    def keyboard_control(self,running):

        for event in pygame.event.get():
           
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:          # check for key presses          
                if event.key == pygame.K_LEFT:        # left arrow turns left
                    self.pressed_left = True
                elif event.key == pygame.K_RIGHT:     # right arrow turns right
                    self.pressed_right = True
                elif event.key == pygame.K_UP:        # up arrow goes up
                    self.pressed_up = True
                elif event.key == pygame.K_DOWN:     # down arrow goes down
                    self.pressed_down = True
                elif event.key == pygame.K_ESCAPE:
                   running = False
            
            elif event.type == pygame.KEYUP:        # check for key releases
                if event.key == pygame.K_LEFT:        # left arrow turns left
                    self.pressed_left = False
                elif event.key == pygame.K_RIGHT:     # right arrow turns right
                    self.pressed_right = False
                elif event.key == pygame.K_UP:        # up arrow goes up
                    self.pressed_up = False
                elif event.key == pygame.K_DOWN:     # down arrow goes down
                    self.pressed_down = False
        if self.pressed_up:
            self.v = self.v+self.a*self.dt
        elif self.pressed_down:
            self.v = self.v-self.a*self.dt
        else:
            self.v = self.v*self.reduce_v
            if(abs(self.v)<self.thres_v):
                self.v=0
        if self.pressed_left:
            self.w = self.w+self.al*self.dt
        elif self.pressed_right:
            self.w = self.w-self.al*self.dt
        else:
            self.w = self.w*self.reduce_w
            if(abs(self.w)<self.thres_w):
                self.w=0
        if(self.v>self.VMAX):
            self.v=self.VMAX
        elif(self.v<-self.VMAX):
            self.v=-self.VMAX
        if(self.w>self.WMAX):
            self.w=self.WMAX
        elif(self.w<-self.WMAX):
            self.w=-self.WMAX
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.v/2,steer=self.w))
        print("Velocity %f and Angular Acceleration %f"%(self.v,self.w))
        print("Velocity of vehicle",self.vehicle.get_velocity())
        return running
        
def main():

    env = CarEnv()


    pygame.init()
    pygame.display.set_mode((100, 100))
    running = True
    while running:
        running = env.keyboard_control(running)
    pygame.quit()
    env.destroy()


main()


