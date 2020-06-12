from model_data_collection_preparation import *

# ==============================================================================
# -- System ID Data Collection ------------------------------------------------
# ==============================================================================



from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_0
from pygame.locals import K_9
from pygame.locals import K_BACKQUOTE
from pygame.locals import K_BACKSPACE
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_c
from pygame.locals import K_g
from pygame.locals import K_d
from pygame.locals import K_h
from pygame.locals import K_m
from pygame.locals import K_n
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_l
from pygame.locals import K_i
from pygame.locals import K_z
from pygame.locals import K_x
from pygame.locals import K_MINUS
from pygame.locals import K_EQUALS

steer = 0
throttle = 0
brake = 0

def parse_vehicle_keys():
    global steer, throttle, brake
    keys = pygame.key.get_pressed()
    pygame.event.pump()

    throttle_increment = 2.5e-1
    if keys[K_UP] or keys[K_w]:
        throttle += throttle_increment
    else:
        throttle = 0.0
    throttle = min(1.0, throttle)

    steer_increment = 2.5e-1
    if keys[K_LEFT] or keys[K_a]:
        # print("K a")
        if steer > 0:
            steer = 0
        else:
            steer -= steer_increment
    elif keys[K_RIGHT] or keys[K_d]:
        # print("K d")
        if steer < 0:
            steer = 0
        else:
            steer += steer_increment
    else:
        steer = 0.0
    steer = min(1, max(-1, steer))

    brake_increment = 2.5e-1
    if keys[K_DOWN] or keys[K_s]:
        brake += brake_increment
    else:
        brake = 0.0
    brake = min(1.0, brake)

    return steer, throttle, brake

systemid_data = []

# carla init
env = CarEnv()
current_state = env.reset()

for _ in tqdm(range(1000)):
    action = list(parse_vehicle_keys())
    
    new_state = env.step(action)

    systemid_data.append(current_state + action + new_state) # (6,), (3,), (6,)

    current_state = new_state

systemid_data = np.array(systemid_data)
# pickle.dump(systemid_data, open("data/systemid_data_100ms_19.data", mode="wb"))
