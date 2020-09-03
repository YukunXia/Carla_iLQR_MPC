from jax import jit, jacfwd, jacrev, hessian, lax
import jax.numpy as np
from jax.scipy.special import logsumexp
import jax

np.set_printoptions(precision=3)

# from jax.config import config
# config.update("jax_enable_x64", True)

import numpy as onp

import pickle
import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = [5, 5]

from tqdm.auto import tqdm

import time

from car_env_for_MPC import *

import os

DT = 0.1# [s] delta time step, = 1/FPS_in_server
N_X = 5
N_U = 3
TIME_STEPS = 50
# MODEL_NAME = "bicycle_model_100000_v2_jax"
MODEL_NAME = "bicycle_model_100ms_20000_v4_jax"
model_path="../SystemID/model/net_{}.model".format(MODEL_NAME)
NN_W1, NN_W2, NN_W3, NN_LR_MEAN = pickle.load(open(model_path, mode="rb"))


@jit
def NN3(x):
    x = np.tanh(NN_W1@x)
    x = np.tanh(NN_W2@x)
    x = NN_W3@x
    
    return x

@jit
def continuous_dynamics(state, u):
    # state = [x, y, v, phi, beta, u]

    x = state[0]
    y = state[1]
    v = state[2]
    v_sqrt = np.sqrt(v)
    phi = state[3]
    beta = state[4]
    steering = np.sin(u[0])
    throttle_brake = np.sin(u[1:])*0.5 + 0.5

    deriv_x = v*np.cos(phi+beta)
    deriv_y = v*np.sin(phi+beta)
    deriv_phi = v*np.sin(beta)/NN_LR_MEAN

    x1 = np.hstack((
                v_sqrt,
                np.cos(beta), 
                np.sin(beta),
                steering,
                throttle_brake
            ))

    x2 = np.hstack((
                v_sqrt,
                np.cos(beta), 
                -np.sin(beta),
                -steering,
                throttle_brake
            ))

    x1 = NN3(x1)
    x2 = NN3(x2)

    deriv_v = ( x1[0]*(2*v_sqrt+x1[0]) + x2[0]*(2*v_sqrt+x2[0]) )/2 # x1[0]+x2[0]
    deriv_beta = ( x1[1] - x2[1] )/2

    derivative = np.hstack((deriv_x, deriv_y, deriv_v/DT, deriv_phi, deriv_beta/DT))

    return derivative

@jit
def discrete_dynamics(state, u):
    return state + continuous_dynamics(state, u)*DT

@jit
def rollout(x0, u_trj):
    x_final, x_trj = jax.lax.scan(rollout_looper, x0, u_trj)
    return np.vstack((x0, x_trj))
    
@jit
def rollout_looper(x_i, u_i):
    x_ip1 = discrete_dynamics(x_i, u_i)
    return x_ip1, x_ip1

# TODO: remove length if weighing func is not needed
@jit
def distance_func(x, route):
    x, ret = lax.scan(distance_func_looper, x, route)
    return -logsumexp(ret)

@jit
def distance_func_looper(input_, p):
    global dp
    
    delta_x = input_[0]-p[0]
    delta_y = input_[1]-p[1]

    return input_, -(delta_x**2.0 + delta_y**2.0)/(1.0*dp**2.0)

@jit
def cost_1step(x, u, route): # x.shape:(5), u.shape(2)
    global TIME_STEPS_RATIO
    steering = np.sin(u[0])
    throttle = np.sin(u[1])*0.5 + 0.5
    brake = np.sin(u[2])*0.5 + 0.5
    
    c_position = distance_func(x, route)
    c_speed = (x[2]-8)**2 # -x[2]**2 
    c_control = (steering**2 + throttle**2 + brake**2 + throttle*brake)

    return (0.04*c_position + 0.002*c_speed + 0.0005*c_control)/TIME_STEPS_RATIO

@jit
def cost_final(x, route): # x.shape:(5), u.shape(2)
    global TARGET_RATIO
    c_position = (x[0]-route[-1,0])**2 + (x[1]-route[-1,1])**2
#     c_position = 0
    c_speed = x[2]**2

    return (c_position/(TARGET_RATIO**2) + 0.0*c_speed)*1

@jit
def cost_trj(x_trj, u_trj, route):
    total = 0.
    total, x_trj, u_trj, route = jax.lax.fori_loop(0, TIME_STEPS-1, cost_trj_looper, [total, x_trj, u_trj, route])
    total += cost_final(x_trj[-1], route)
    
    return total

# XXX: check if the cost_1step needs `target`
@jit
def cost_trj_looper(i, input_):
    total, x_trj, u_trj, route = input_
    total += cost_1step(x_trj[i], u_trj[i], route)
    
    return [total, x_trj, u_trj, route]

@jit
def derivative_init():
    jac_l = jit(jacfwd(cost_1step, argnums=[0,1]))
    hes_l = jit(hessian(cost_1step, argnums=[0,1]))
    jac_l_final = jit(jacfwd(cost_final))
    hes_l_final = jit(hessian(cost_final))
    jac_f = jit(jacfwd(discrete_dynamics, argnums=[0,1]))
    
    return jac_l, hes_l, jac_l_final, hes_l_final, jac_f

jac_l, hes_l, jac_l_final, hes_l_final, jac_f = derivative_init()

@jit
def derivative_stage(x, u, route): # x.shape:(5), u.shape(3)
    global jac_l, hes_l, jac_f
    l_x, l_u = jac_l(x, u, route)
    (l_xx, l_xu), (l_ux, l_uu) = hes_l(x, u, route)
    f_x, f_u = jac_f(x, u)

    return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

@jit
def derivative_final(x, target):
    global jac_l_final, hes_l_final
    l_final_x = jac_l_final(x, target)
    l_final_xx = hes_l_final(x, target)

    return l_final_x, l_final_xx

@jit
def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    Q_x = l_x + f_x.T@V_x
    Q_u = l_u + f_u.T@V_x

    Q_xx = l_xx + f_x.T@V_xx@f_x
    Q_ux = l_ux + f_u.T@V_xx@f_x
    Q_uu = l_uu + f_u.T@V_xx@f_u

    return Q_x, Q_u, Q_xx, Q_ux, Q_uu

@jit
def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = - Q_uu_inv@Q_u
    K = - Q_uu_inv@Q_ux

    return k, K

@jit
def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    V_x = Q_x + K.T@Q_u + Q_ux.T@k + K.T@Q_uu@k
    V_xx = Q_xx + 2*K.T@Q_ux + K.T@Q_uu@K

    return V_x, V_xx

@jit
def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T@k - 0.5 * k.T@Q_uu@k

@jit
def forward_pass(x_trj, u_trj, k_trj, K_trj):
    u_trj = np.arcsin(np.sin(u_trj))
    
    x_trj_new = np.empty_like(x_trj)
    x_trj_new = jax.ops.index_update(x_trj_new, jax.ops.index[0], x_trj[0])
    u_trj_new = np.empty_like(u_trj)
    
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = lax.fori_loop(
        0, TIME_STEPS-1, forward_pass_looper, [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]
    )

    return x_trj_new, u_trj_new

@jit
def forward_pass_looper(i, input_):
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = input_
    
    u_next = u_trj[i] + k_trj[i] + K_trj[i]@(x_trj_new[i] - x_trj[i])
    u_trj_new = jax.ops.index_update(u_trj_new, jax.ops.index[i], u_next)

    x_next = discrete_dynamics(x_trj_new[i], u_trj_new[i])
    x_trj_new = jax.ops.index_update(x_trj_new, jax.ops.index[i+1], x_next)
    
    return [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]

@jit
def backward_pass(x_trj, u_trj, regu, target):
    k_trj = np.empty_like(u_trj)
    K_trj = np.empty((TIME_STEPS-1, N_U, N_X))
    expected_cost_redu = 0.
    V_x, V_xx = derivative_final(x_trj[-1], target)
     
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target = lax.fori_loop(
        0, TIME_STEPS-1, backward_pass_looper, [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target]
    )
        
    return k_trj, K_trj, expected_cost_redu


@jit
def backward_pass_looper(i, input_):
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target = input_
    n = TIME_STEPS-2-i
    
    l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivative_stage(x_trj[n], u_trj[n], target)
    Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
    Q_uu_regu = Q_uu + np.eye(N_U)*regu
    k, K = gains(Q_uu_regu, Q_u, Q_ux)
    k_trj = jax.ops.index_update(k_trj, jax.ops.index[n], k)
    K_trj = jax.ops.index_update(K_trj, jax.ops.index[n], K)
    V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
    expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    
    return [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target]

@jit
def run_ilqr_main(x0, u_trj, target):
    global jac_l, hes_l, jac_l_final, hes_l_final, jac_f
    
    max_iter=300
    regu = np.array(100.)
    
    x_trj = rollout(x0, u_trj)
    cost_trace = jax.ops.index_update(
        np.zeros((max_iter+1)), jax.ops.index[0], cost_trj(x_trj, u_trj, target)
    )

    x_trj, u_trj, cost_trace, regu, target = lax.fori_loop(
        1, max_iter+1, run_ilqr_looper, [x_trj, u_trj, cost_trace, regu, target]
    )
    
    return x_trj, u_trj, cost_trace

@jit
def run_ilqr_looper(i, input_):
    x_trj, u_trj, cost_trace, regu, target = input_
    k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu, target)
    x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj)
    
    total_cost = cost_trj(x_trj_new, u_trj_new, target)
    
    x_trj, u_trj, cost_trace, regu = lax.cond(
        pred = (cost_trace[i-1] > total_cost),
        true_operand = [i, cost_trace, total_cost, x_trj, u_trj, x_trj_new, u_trj_new, regu],
        true_fun = run_ilqr_true_func,
        false_operand = [i, cost_trace, x_trj, u_trj, regu],
        false_fun = run_ilqr_false_func,
    )
    
    max_regu = 10000.0
    min_regu = 0.01
    
    regu += jax.nn.relu(min_regu - regu)
    regu -= jax.nn.relu(regu - max_regu)

    return [x_trj, u_trj, cost_trace, regu, target]

@jit
def run_ilqr_true_func(input_):
    i, cost_trace, total_cost, x_trj, u_trj, x_trj_new, u_trj_new, regu = input_
    
    cost_trace = jax.ops.index_update(
        cost_trace, jax.ops.index[i], total_cost 
    )
    x_trj = x_trj_new
    u_trj = u_trj_new
    regu *= 0.7
    
    return [x_trj, u_trj, cost_trace, regu]

@jit
def run_ilqr_false_func(input_):
    i, cost_trace, x_trj, u_trj, regu = input_
    
    cost_trace = jax.ops.index_update(
        cost_trace, jax.ops.index[i], cost_trace[i-1] 
    )
    regu *= 2.0
    
    return [x_trj, u_trj, cost_trace, regu]


TIME_STEPS = 60

# NOTE: Set dp to be the same as carla
dp = 1 # same as waypoint interval

onp.random.seed(1)


TIME_STEPS_RATIO = TIME_STEPS/50
# TARGET_RATIO = np.linalg.norm(target[-1]-target[0])/(3*np.pi)
TARGET_RATIO = FUTURE_WAYPOINTS_AS_STATE*dp/(6*np.pi) # TODO: decide if this should be determined dynamically


# carla init
env = CarEnv()
for i in range(1):
    state, waypoints = env.reset()

    # total_time = 0

    for k in tqdm(range(2000)):
        # start = time.time()

        state[2] += 0.01
        state = np.array(state)
        
        u_trj = onp.random.randn(TIME_STEPS-1, N_U)*1e-8
        u_trj[:,2] -= np.pi/2.5
        # u_trj[:,1] -= np.pi/8
        u_trj = np.array(u_trj)
        
        waypoints = np.array(waypoints)
        
        x_trj, u_trj, cost_trace = run_ilqr_main(state, u_trj, waypoints)

        # end = time.time()
        # if k > 1:
        #     total_time += end - start
        
        draw_planned_trj(env.world, onp.array(x_trj), env.location_[2], color=(0, 223, 222))

        for j in range(MPC_INTERVAL):
            steering = np.sin(u_trj[j,0])
            throttle = np.sin(u_trj[j,1])*0.5 + 0.5
            brake = np.sin(u_trj[j,2])*0.5 + 0.5
            state, waypoints, done, _ = env.step(onp.array([steering, throttle, brake]))

        # tqdm.write("final estimated cost = {0:.2f} \n velocity = {1:.2f}".format(cost_trace[-1],state[2]))
        # if k > 1:
        #     tqdm.write("mean MPC calc time = {}".format(total_time/(k)))

        # if done:
        #     break

pygame.quit()

if VIDEO_RECORD:
    os.system("ffmpeg -r 50 -f image2 -i Snaps/%05d.png -s {}x{} -aspect 16:9 -vcodec libx264 -crf 25 -y Videos/result.avi".
                format(RES_X, RES_Y))



# TODO:
# * check 0 speed issue
# * optimize speed cost setting
# * retrain dynamical model, perhaps with automatically collected data and boosting/weighing
# * check uncertainty model and iLQG
# * check Guided Policy Search