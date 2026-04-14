import torch
from config import eegnet
from models import EEGNet
import numpy as np




x = torch.rand(1, 1, 64, 128)

model = EEGNet(config=eegnet)
x = model(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model=model)) # should be ~2.963 params