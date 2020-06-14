''' 
MIT License

Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
Barcelona (UAB).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import random
import time
from enum import Enum # for RoadOption
import math

from tqdm.auto import tqdm
import pickle


try:
    import queue
except ImportError:
    import Queue as queue



import carla


try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# -- Define constants ---------------------------------------------------------
# ==============================================================================

DT_ = 0.01 # [s] delta time step, = 1/FPS_in_server
N_DT = 10 # 10 ticks -> 100ms as time interval for data collection

NO_RENDERING = False

SHOW_CAM = True 

START_TIME = 0

# ==============================================================================
# -- Predefine ---------------------------------------------------------
# ==============================================================================

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


class RoadOption(Enum): # for waypoint setting
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)



# ==============================================================================
# -- Main Class ---------------------------------------------------------
# ==============================================================================

class CarEnv:

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.world = self.client.reload_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.vehicle = None

        # world in sync mode
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=NO_RENDERING, # False for debug
            synchronous_mode=True,
            fixed_delta_seconds=DT_))

        pygame.init()
        self.display = pygame.display.set_mode(
                (800, 600),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()


    def reset(self): # reset at the beginning of each episode "current_state = env.reset()"
        # print("call reset. ")
        tqdm.write("call reset")
        self.collision_hist = []
        self.actor_list = [] # include the vehicle and the collision sensor # no multiagent at this point

        self.transform = random.choice(self.world.get_map().get_spawn_points()) 
        new_car = False
        if self.vehicle is None:
            new_car = True
        # print("new_car =", new_car)
        
        if new_car == True:
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            self.actor_list.append(self.vehicle)
        else:
            self.vehicle.set_transform(self.transform)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        if new_car == True:
            transform = carla.Transform(carla.Location(x=2.5, z=0.7)) # transform for collision sensor attachment
            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event)) # what's the mechanism?
            
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
            time.sleep(4)

            self.camera = self.world.spawn_actor(
                self.blueprint_library.find('sensor.camera.rgb'),
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.actor_list.append(self.camera)
            self.image_queue = queue.Queue()
            self.camera.listen(self.image_queue.put)

            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))


        self.time = 0
        return self.get_state()

    def collision_data(self, event):
        self.collision_hist.append(event)
    
    def get_state(self,):
        state = []

        # # collect information

        self.location = self.vehicle.get_location()
        self.location_ = np.array([self.location.x, self.location.y, self.location.z])
        self.velocity = self.vehicle.get_velocity()
        self.acceleration = self.vehicle.get_acceleration()
        self.transform = self.vehicle.get_transform()
        self.yaw = self.transform.rotation.yaw
        self.angular_velocity = self.vehicle.get_angular_velocity()

        state = [self.location.x, self.location.y]
        state += [self.velocity.x, self.velocity.y]
        state += [self.acceleration.x, self.acceleration.y]
        state += [self.yaw]
        state += [self.angular_velocity.z]

        return state

    def step(self, action): # 0:steer; 1:throttle; 2:brake; np array shape = (3,)
        assert len(action) == 3

        if self.time >= START_TIME: # starting time
            steer_, throttle_, brake_ = action
        else:
            steer_ = 0
            throttle_ = 0
            brake_ = 1

        tqdm.write("steer_ = {0:5.2f}, throttle_ {1:5.2f}, brake_ {2:5.2f}".format(float(steer_), float(throttle_), float(brake_)))
        assert steer_ >= -1 and steer_ <= 1 and throttle_ <= 1 and throttle_ >= 0 and  brake_ <= 1 and brake_ >= 0

        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle_), steer=float(steer_), brake=float(brake_)))

        for _ in range(N_DT):
            self.clock.tick()
            self.world.tick() 
            self.time += DT_
            image_rgb = self.image_queue.get()
            draw_image(self.display, image_rgb)
            self.display.blit(
                self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                (8, 10))
            self.display.blit(
                self.font.render('% 5d FPS (simulated)' % int(1/DT_), True, (255, 255, 255)),
                (8, 28))
            pygame.display.flip()


        new_state = self.get_state()
        
        return new_state