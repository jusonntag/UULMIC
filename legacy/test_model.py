from models.config import eegnet
from models import EEGNet
import numpy as np
from pathlib import Path

import torch
print(torch.cuda.is_available())

data_dir_x = Path.cwd() / 'data' / 'preprocessed' / 'EEGNet' / '10' / '10_EEGNet_X_4.0_40.0_4.5_ICA_128.npy'
data_dir_y = Path.cwd() / 'data' / 'preprocessed' / 'EEGNet' / '10' / '10_EEGNet_Y_4.0_40.0_4.5_ICA_128.npy'

# load data
X = np.load(data_dir_x)
Y = np.load(data_dir_y)

# add dim -> (batch, 1, channel, samples)
X = torch.from_numpy(np.expand_dims(X, axis = 1)).float()

# set samples to the number of samples form the real data
eegnet.samples = X.shape[-1]

# configurate model
model = EEGNet(config=eegnet)

# pass data to model
x = model(X)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model=model)) # should be ~2.963 params | got 2.739 params 29.03.26