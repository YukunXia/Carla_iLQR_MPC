from collections import deque # for waypoint recording

import numpy as onp
import random
from enum import Enum # for RoadOption
import math

from tqdm.auto import tqdm


try:
    import queue
except ImportError:
    import Queue as queue
import itertools

import carla


try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# -- Define constants ---------------------------------------------------------
# ==============================================================================

DT_ = 0.01 # [s] delta time step, = 1/FPS_in_server
N_DT = 10 # 5 ticks for training DDPG once

NO_RENDERING = False

WAYPOINT_BUFFER_LEN = 100
WAYPOINT_INTERVAL = 1 # [m]
WAYPOINT_BUFFER_MID_INDEX = int(WAYPOINT_BUFFER_LEN/2)

# BETA_HISTORY_LEN = 15

FUTURE_WAYPOINTS_AS_STATE = 50

SHOW_CAM = True 

START_TIME = 3

DEBUG = True

MPC_INTERVAL = 1

# ==============================================================================
# -- Predefininition ---------------------------------------------------------
# ==============================================================================

def draw_image(surface, image, blend=False):
    array = onp.frombuffer(image.raw_data, dtype=onp.dtype("uint8"))
    array = onp.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def draw_waypoints(world, waypoints, z=0.5, color=(255,0,0)): # from carla/agents/tools/misc.py
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    color = carla.Color(r=color[0],g=color[1],b=color[2],a=255)
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.05, color=color, life_time=0.1)

def draw_planned_trj(world, x_trj, car_z, color=(255,0,0)):
    color = carla.Color(r=color[0],g=color[1],b=color[2],a=255)
    length = x_trj.shape[0]
    xx = x_trj[:,0]
    yy = x_trj[:,1]
    for i in range(1, length):
        begin = carla.Location(float(xx[i-1]), float(yy[i-1]), float(car_z+1))
        end = carla.Location(float(xx[i]), float(yy[i]), float(car_z+1))
        # thickness = 
        world.debug.draw_line(begin=begin, end=end, thickness=0.1, color=color, life_time=0.1*MPC_INTERVAL)


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

# two functions for waypoint selection

def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options

def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

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
        # self.lr = 1.538900111477258 # from system identification
        self.vehicle = None
        self.actor_list = []

        # world in sync mode
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=NO_RENDERING, # False for debug
            synchronous_mode=True,
            fixed_delta_seconds=DT_))

        # waypoint init
        self.waypoint_buffer = deque(maxlen=WAYPOINT_BUFFER_LEN)

        # # same format as OpenAI gym
        # self.nb_states = 5
        # self.nb_actions = 3

        pygame.init()
        self.display = pygame.display.set_mode(
                (800, 600),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()

        # self.waypoint_chasing_index = 0

    def reset(self): # reset at the beginning of each episode "current_state = env.reset()"
        # print("call reset. ")
        tqdm.write("call reset")


        self.collision_hist = []
        self.actor_list = [] # include the vehicle and the collision sensor # no multiagent at this point

        self.spawn_point = random.choice(self.world.get_map().get_spawn_points()) # everytime set a new spawning point # ??? How about the destination?
        #self.spawn_point = self.world.get_map().get_spawn_points()[8] # fixed for testing

        new_car = False
        if self.vehicle is None:
            new_car = True

        if new_car == True:
            self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
            self.actor_list.append(self.vehicle)
        else:
            self.vehicle.set_transform(self.spawn_point)

        if new_car == True:
            transform = carla.Transform(carla.Location(x=2.5, z=0.7)) # transform for collision sensor attachment
            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event)) # what's the mechanism?
            
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

            self.camera = self.world.spawn_actor(
                self.blueprint_library.find('sensor.camera.rgb'),
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.actor_list.append(self.camera)
            self.image_queue = queue.Queue()
            self.camera.listen(self.image_queue.put)

            # # self.episode_start = time.time() # comment it bcs env is in sync mode
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        self.waypoint_buffer = deque(maxlen=WAYPOINT_BUFFER_LEN)
        # self.waypoint_chasing_index = 0
        self.update_waypoint_buffer(given_loc=[True, self.spawn_point.location])

        # self.beta_queue = deque(maxlen=BETA_HISTORY_LEN)
        # self.beta_queue.append(0.0)
        self.time = 0
        return self.get_state(), self.get_waypoint()

    def collision_data(self, event):
        self.collision_hist.append(event)

    def update_waypoint_buffer(self, given_loc = [False, None]):
        if given_loc[0]:
            car_loc = given_loc[1]
        else:
            car_loc = self.vehicle.get_location()

        self.min_distance = onp.inf
        if (len(self.waypoint_buffer) == 0):
            self.waypoint_buffer.append(self.map.get_waypoint(car_loc))

        for i in range(len(self.waypoint_buffer)):
            curr_distance = self.waypoint_buffer[i].transform.location.distance(car_loc)
            if curr_distance < self.min_distance:
                self.min_distance = curr_distance
                min_distance_index = i

        num_waypoints_to_be_added = max(0, min_distance_index - WAYPOINT_BUFFER_MID_INDEX)
        num_waypoints_to_be_added = max(num_waypoints_to_be_added, WAYPOINT_BUFFER_LEN - len(self.waypoint_buffer))

        for _ in range(num_waypoints_to_be_added):
            frontier = self.waypoint_buffer[-1]
            next_waypoints = list(frontier.next(WAYPOINT_INTERVAL))
            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, frontier)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]
            self.waypoint_buffer.append(next_waypoint)
        
        self.min_distance_index = WAYPOINT_BUFFER_MID_INDEX if min_distance_index > WAYPOINT_BUFFER_MID_INDEX else min_distance_index
        # self.waypoint_chasing_index = max(self.waypoint_chasing_index, self.min_distance_index+1)
    
    def get_state(self,):

        # collect information

        self.location = self.vehicle.get_location()
        self.location_ = onp.array([self.location.x, self.location.y, self.location.z])

        # tqdm.write("get state, loc = {}, spawn = {}".format(self.location_[:2], [self.spawn_point.location.x, self.spawn_point.location.y]))

        self.transform = self.vehicle.get_transform()
        # self.yaw = onp.array(self.transform.rotation.yaw) # float, only yaw: only along z axis # check https://d26ilriwvtzlb.cloudfront.net/8/83/BRMC_9.jpg 
        phi = self.transform.rotation.yaw*onp.pi/180 # phi is yaw

        self.velocity = self.vehicle.get_velocity()
        vx = self.velocity.x
        vy = self.velocity.y

        beta_candidate = onp.arctan2(vy, vx) - phi + onp.pi*onp.array([-2,-1,0,1,2])
        local_diff = onp.abs(beta_candidate - 0)
        min_index = onp.argmin(local_diff)
        beta = beta_candidate[min_index]

        # state = [self.velocity.x, self.velocity.y, self.yaw, self.angular_velocity.z]
        state = [
                    self.location.x, # x
                    self.location.y, # y
                    onp.sqrt(vx**2 + vy**2), # v
                    phi, # phi
                    beta, # beta
                ]

        return onp.array(state)

    def get_waypoint(self,):
        waypoints = []
        for i in range(self.min_distance_index, self.min_distance_index+FUTURE_WAYPOINTS_AS_STATE):
            waypoint_location = self.waypoint_buffer[i].transform.location
            waypoints.append([waypoint_location.x, waypoint_location.y])

        return onp.array(waypoints)

    def step(self, action): # 0:steer; 1:throttle; 2:brake; onp array shape = (3,)
        assert len(action) == 3

        if self.time >= START_TIME: # starting time
            steer_, throttle_, brake_ = action
        else:
            steer_ = 0
            throttle_ = 0.5
            brake_ = 0

        assert steer_ >= -1 and steer_ <= 1 and throttle_ <= 1 and throttle_ >= 0 and  brake_ <= 1 and brake_ >= 0

        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle_), steer=float(steer_), brake=float(brake_)))

        # move a step
        for _ in range(N_DT):
            self.clock.tick()
            self.world.tick() # needs to be tested! use time.sleep(??) to test
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

        # do we need to wait for tick (10) first?
        # no! wait_for_tick() + world.on_tick works for async mode
        # check https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#simulation-time-step
        
        self.update_waypoint_buffer()
        if DEBUG:
            past_WP = list(itertools.islice(self.waypoint_buffer, 0, self.min_distance_index))
            future_WP = list(itertools.islice(self.waypoint_buffer, self.min_distance_index+1, WAYPOINT_BUFFER_LEN-1))
            draw_waypoints(self.world, future_WP, z=self.location.z+0.5, color=(255,0,0))
            draw_waypoints(self.world, past_WP, z=self.location.z+0.5, color=(0,255,0))
            draw_waypoints(self.world, [self.waypoint_buffer[self.min_distance_index]], z=self.location.z+0.5, color=(0,0,255))
            # draw_waypoints(self.world, self.waypoint_buffer)

        if len(self.collision_hist) != 0:
            done = True
        else:
            done = False

        new_state = self.get_state()
        waypoints = self.get_waypoint()

        return new_state, waypoints, done, None # new_state: onp array, shape = (N,)



# # ==============================================================================
# # -- Testing ---------------------------------------------------------
# # ==============================================================================

# if __name__ == "__main__":

#     # carla init
#     env = CarEnv()
#     for i in range(1):
#         state = env.reset()

#         # while True:
#         for _ in tqdm(range(200)):

#             # action = f(current_State)
#             action = onp.random.rand(3)
#             action[0] = (action[0]-0.5)*2
#             action[1] = 1
#             action[2] = 0
            
#             state, waypoints, done, _ = env.step(action)

#             x_trj = onp.vstack((onp.linspace(0,10,11), onp.zeros(11))).T
#             x_trj += env.location_[:2]
#             draw_planned_trj(env.world, x_trj, env.location_[2], color=(0, 223, 222))

#             tqdm.write("state = {}".format(state))

#             if done:
#                 break
