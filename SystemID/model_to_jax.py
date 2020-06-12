import torch
import torch.nn.functional as F

import pickle

import numpy as np

from system_id import Net_v4

TRAINING_NAME = "bicycle_model_100ms_20000_v4"

model_path = "model/net_{}.model".format(TRAINING_NAME)
model = torch.load(model_path).to("cpu")

w1, w2, w3 = model.parameters()
w1 = w1.detach().numpy()
w2 = w2.detach().numpy()
w3 = w3.detach().numpy()

lr_mean_path = "param/lr_mean"
lr_mean = pickle.load(open(lr_mean_path, mode="rb"))

pickle.dump([w1,w2,w3,lr_mean], open("model/net_{}_jax.model".format(TRAINING_NAME), mode="wb"))