'''
Reference for vehicle model:

https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf
'''

import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use("default")

from tqdm.auto import tqdm

from scipy.ndimage import gaussian_filter1d

START_INDEX = 50
BATCH_SIZE = 1024
CUDA = torch.cuda.is_available()


# ==============================================================================
# -- Preprocess data ---------------------------------------------------------
# ==============================================================================

model_data_ = np.vstack(([pickle.load(open("data/systemid_data_100ms_{}.data".format(i), mode="rb"))[START_INDEX:,:] for i in range(20)]))
model_data = np.hstack((model_data_[:,:4], model_data_[:,6:15], model_data_[:,17:])) # exclude acceleration

model_data[:,4:6] *= np.pi/180 # angle unit transforms from degree to radian
model_data[:,-2:] *= np.pi/180

# name_ls = [
#     ["x   ", "y   ", "vx  ", "vy  ", "yaw ", "avz "],
#     ["stee", "thro", "brak"]
# ]


vy = model_data[:,3]
vy_ = model_data[:,-3]

vx = model_data[:,2]
vx_ = model_data[:,-4]

v = np.sqrt(vx**2 + vy**2)
# plt.hist(v, 100)
v_ = np.sqrt(vx_**2 + vy_**2)

v_sqrt = np.sqrt(v)
v_sqrt_ = np.sqrt(v_)

yaw = model_data[:,4]
yaw_ = model_data[:,-2]

avz = model_data[:,5]
avz_ = model_data[:,-1]

beta = np.arctan2(vy, vx) - yaw
beta_ = np.arctan2(vy_, vx_) - yaw_

beta_revised = [beta[0]]
beta_revised_ = [beta_[0]]
for i in range(1, beta.size):
    beta_candidate = beta[i] + np.pi*np.array([-2,-1,0,1,2])
    local_diff = np.abs(beta_candidate - 0)
    min_index = np.argmin(local_diff)
    beta_revised.append(beta_candidate[min_index])

    beta_candidate_ = beta_[i] + np.pi*np.array([-2,-1,0,1,2])
    local_diff_ = np.abs(beta_candidate_ - 0)
    min_index_ = np.argmin(local_diff_)
    beta_revised_.append(beta_candidate_[min_index_])

beta_revised = np.array(beta_revised)
beta_revised_smoothed = gaussian_filter1d(beta_revised, 1)
beta_revised_ = np.array(beta_revised_)

delta_beta_revised = beta_revised_[:] - beta_revised[:]
delta_beta_revised_smoothed = gaussian_filter1d(delta_beta_revised, 1)

lr = [] # a length used in the vehicle model, check the link at top
for i in range(beta.size):
    cache = v[i]*np.sin(beta_revised_smoothed[i])/avz[i]
    if avz[i] > 0.1 and cache > 0:
        lr.append(cache)

lr = np.array(lr)
lr_mean = lr.mean() # approximate 1.5-1.6 [m]
pickle.dump(lr_mean, open("param/lr_mean", mode="wb"))

delta_v = v_ - v
delta_v_sqrt = v_sqrt_ - v_sqrt

steering = model_data[:,6]
throttle = model_data[:,7]
brake = model_data[:,8]

# feed data in to pytorch

input_ = np.zeros((beta.size, 6))
for i, element in enumerate([v, beta_revised, steering, throttle, brake]):
    input_[:,i] = element

prediction = np.zeros((beta.size, 2))
prediction[:,0] = delta_v
prediction[:,1] = delta_beta_revised_smoothed

input_ = torch.FloatTensor(input_)
prediction = torch.FloatTensor(prediction)

training_size = int(input_.shape[0]*0.8)
test_size = int(input_.shape[0]*0.2)

dataset = Data.TensorDataset(input_, prediction)
training_set, test_set = Data.random_split(dataset, [training_size, test_size])
training_loader = Data.DataLoader(training_set, batch_size=BATCH_SIZE)
test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE)


# ==============================================================================
# -- Define NN v4 ---------------------------------------------------------
# ==============================================================================


class Net_v4(torch.nn.Module):
    def __init__(self, n_feature=5, n_hidden=64, n_output=2):
        super(Net_v4, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature+1, n_hidden, bias=False)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden, bias=False)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)   # output layer

    def forward(self, input_, training=False):
        v_sqrt = torch.sqrt(input_[:,0:1])
        beta = input_[:,1:2]
        steering = input_[:,2:3]
        throttle_brake = input_[:,3:5]
        # brake = input_[:,4:5]

        x1 = torch.cat(
            (
                v_sqrt, 
                torch.cos(beta), 
                torch.sin(beta),
                steering,
                throttle_brake
            ),
            dim = -1
        )

        x2 = torch.cat(
            (
                v_sqrt, 
                torch.cos(beta), 
                -torch.sin(beta),
                -steering,
                throttle_brake
            ),
            dim = -1
        )

        x1 = F.tanh(self.hidden1(x1)) # model with LeakyRelu has smaller MSE, but the model is nonsmooth
        x1 = F.tanh(self.hidden2(x1)) # iLQR prefers smooth model
        x1 = self.predict(x1)            
        v_sqrt_dot1 = x1[:,0].unsqueeze(1) # not the real gradient of v_sqrt
        beta_dot1 = x1[:,1].unsqueeze(1)

        x2 = F.tanh(self.hidden1(x2))     
        x2 = F.tanh(self.hidden2(x2))  
        x2 = self.predict(x2)            
        v_sqrt_dot2 = x2[:,0].unsqueeze(1)
        beta_dot2 = x2[:,1].unsqueeze(1)

        x = torch.cat( # enforce the symmetry of model, and the positivity of "v"
            (
                v_sqrt_dot1*(2*v_sqrt+v_sqrt_dot1) + v_sqrt_dot2*(2*v_sqrt+v_sqrt_dot2), 
                # not the real gradient of v, but this form guarantees v to be positive
                beta_dot1 - beta_dot2
            ), 
            dim = -1
        )/2

        return x # predicts $\Delta x$, i.e. the residual, rather than the gradient


TRAINING_NAME = "bicycle_model_100ms_20000_v4"
writer = SummaryWriter()

net_v4 = Net_v4(n_feature=5, n_hidden=32, n_output=2)
if CUDA:
    net_v4.to("cuda")

optimizer = torch.optim.Adam(net_v4.parameters(), lr=1e-4)

# training_history = []
# test_history = []

net_v4.eval()
with torch.no_grad():
    training_loss = 0
    for data,target in training_loader:
        data = data.to("cuda")
        target = target.to("cuda")
        output = net_v4(data, training=True)
        training_loss += F.mse_loss(output, target, reduction="sum").item()
        # if CUDA:
        #     training_loss = training_loss.cpu()
    tqdm.write("epoch = {:3}, avg training loss = {:.6f}".format(0, training_loss/training_size))
    # training_history.append(training_loss)

    test_loss = 0
    for data,target in test_loader:
        data = data.to("cuda")
        target = target.to("cuda")
        output = net_v4(data, training=True)
        test_loss += F.mse_loss(output, target, reduction="sum").item()
    tqdm.write("epoch = {:3}, avg test loss = {:.6f}".format(0, test_loss/test_size))
    # test_history.append(test_loss)


for t in tqdm(range(3000)):
    net_v4.train()
    for data,target in training_loader:
        data = data.to("cuda")
        target = target.to("cuda")
        optimizer.zero_grad()
        output = net_v4(data, training=True)
        training_loss = F.mse_loss(output, target, reduction="mean")
        training_loss.backward()
        optimizer.step()
    if t%20 == 19:
        tqdm.write("epoch = {:3}, training loss = {:.6f}".format(t+1, training_loss.item()))
        # print("epoch = {:3}, avg training loss = {:.6f}".format(t, training_loss/len(training_loader.dataset)))
    # training_history.append(training_loss.item())

    net_v4.eval()
    test_loss = 0
    with torch.no_grad():
        for data,target in test_loader:
            data = data.to("cuda")
            target = target.to("cuda")
            output = net_v4(data, training=True)
            test_loss += F.mse_loss(output, target, reduction="sum").item()
    if t%20 == 19:
        tqdm.write("epoch = {:3}, avg test loss = {:.6f}".format(t+1, test_loss/test_size))
        # print("epoch = {:3}, avg test loss = {:.6f}".format(t, test_loss)
    # test_history.append(test_loss/test_size)

    writer.add_scalars("system identification regression loss - {}".format(TRAINING_NAME), {
        "training": training_loss.item(),
        "test": test_loss/test_size
    }, t)

torch.save(net_v4, "NN_model/net_{}.model".format(TRAINING_NAME))
# pickle.dump(training_history, open("log/training{}.history".format(TRAINING_NAME), mode="wb"))
# pickle.dump(test_history, open("log/test{}.history".format(TRAINING_NAME), mode="wb"))

writer.close()

model_path="obj/net_{}.model".format(TRAINING_NAME)
net_v4 = torch.load(model_path).to("cpu")

net_v4.to("cpu")
net_v4.eval()


# ==============================================================================
# -- dynamics ---------------------------------------------------------
# ==============================================================================

def continuous_dynamics(state, u):
    # state = xu[:]
    # state = [[x, y, v, phi, beta], ...]
    global net_v4
    global lr_mean
    if len(state.shape) == 1:
        state = state.unsqueeze(0)
    
    # with torch.no_grad():

    x = state[:,0:1]
    y = state[:,1:2]
    phi = state[:,3:4]
    v = state[:,2:3]
    beta = state[:,4:5]

    input_ = torch.cat((v, beta, u), dim=-1)

    delta_x = v*torch.cos(phi+beta)
    delta_y = v*torch.sin(phi+beta)
    delta_phi = v*torch.sin(beta)/lr_mean

    delta_v_beta = net_v4(input_)/0.1
    delta_v = delta_v_beta[:,0].unsqueeze(1)
    delta_beta = delta_v_beta[:,1].unsqueeze(1)

    derivative = torch.cat((delta_x, delta_y, delta_v, delta_phi, delta_beta), dim=-1)

    return derivative

def discrete_dynamics(state, u):
    state += 0.1*continuous_dynamics(state,u)
    return state

# ==============================================================================
# -- testing ---------------------------------------------------------
# ==============================================================================

# Test case 1: rotating

state = torch.zeros((1,5))
state[0,3] = 0
u = torch.zeros((1,3))
u[:,0] = -1 # steering = left max
u[:,1] = 1 # throttle  = max

x_trj = [float(state[0,0].numpy())]
y_trj = [float(state[0,1].numpy())]
v_trj = [float(state[0,2].numpy())]
phi_trj = [float(state[0,3].numpy())]
beta_trj = [float(state[0,4].numpy())]
for i in tqdm(range(100)):
    state = discrete_dynamics(state, u).detach()
    x_trj.append(float(state[0,0].numpy()))
    y_trj.append(float(state[0,1].numpy()))
    v_trj.append(float(state[0,2].numpy()))
    phi_trj.append(float(state[0,3].numpy()))
    beta_trj.append(float(state[0,4].numpy()))

veryify_circle_data = pickle.load(open("data/systemid_data_verify_circle.data", mode="rb"))

x_trj_cir = veryify_circle_data[:,0]
y_trj_cir = veryify_circle_data[:,1]
vx_cir = veryify_circle_data[:,2]
vy_cir = veryify_circle_data[:,3]
v_cir = np.sqrt(vx_cir**2 + vy_cir**2)
throttle_cir = veryify_circle_data[:,9]
yaw_cir = veryify_circle_data[:,4]
state_cir = veryify_circle_data[500,:8]

# compare the following two curves
# plt.plot(v_cir[274:-1:10]) # interval = 10 because that data was collected at 10ms time interval
# plt.plot(v_trj[:])

# Test case 2: straight moving

state = torch.zeros((1,5))
state[0,3] = 0.5*np.pi
u = torch.zeros((1,3)) # brake = 0
u[:,0] = 0 # steering = 0
u[:,1] = 1 # throttle = max

x_trj = [float(state[0,0].numpy())]
y_trj = [float(state[0,1].numpy())]
v_trj = [float(state[0,2].numpy())]
phi_trj = [float(state[0,3].numpy())]
beta_trj = [float(state[0,4].numpy())]
for i in tqdm(range(100)):
    state = discrete_dynamics(state, u).detach()
    x_trj.append(float(state[0,0].numpy()))
    y_trj.append(float(state[0,1].numpy()))
    v_trj.append(float(state[0,2].numpy()))
    phi_trj.append(float(state[0,3].numpy()))
    beta_trj.append(float(state[0,4].numpy()))

veryify_straight_data = pickle.load(open("data/systemid_data_verify_straight.data", mode="rb"))

x_trj__ = veryify_straight_data[:,0]
y_trj__ = veryify_straight_data[:,1]
vx__ = veryify_straight_data[:,2]
vy__ = veryify_straight_data[:,3]
v__ = np.sqrt(vx__**2 + vy__**2)
throttle__ = veryify_straight_data[:,9]

# compare the following two curves
# plt.plot(v__[325:1325:10]) # interval = 10 because that data was collected at 10ms time interval
# plt.plot(v_trj[:])

