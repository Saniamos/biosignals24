import torch
from torch import nn

name = "LinearRegression"

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear(x)
        return out
    
    
hyper_params_default = {
    "name": f"{name}_default",
    "epochs": 10,
    "input_size": 848 * 480,
    "output_size": 126,
    "lr_decay_milestones": [0],
    "lr_decay_gamma": 1.0,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": True,
    "shuffle": True,
    "batch_size": 64,
    "normalize": False,
    "description":"""
Eine einfache lineare Regression
"""
}    
    
hyper_params_no_rot = {
    "name": f"{name}_no_rot",
    "epochs": 100,
    "input_size": 848 * 480,
    "output_size": 63,
    "lr_decay_milestones": [0],
    "lr_decay_gamma": 1.0,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": False,
    "shuffle": True,
    "batch_size": 64,
    "normalize": False,
    "description":"""
Eine einfache lineare Regression ohne Rotation
"""
}    


hyper_params_joints = {
    "name": f"{name}_joints",
    "epochs": 10,
    "input_size": 848 * 480,
    "output_size": 60,
    "lr_decay_milestones": [0],
    "lr_decay_gamma": 1.0,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": False,
    "shuffle": True,
    "batch_size": 64,
    "normalize": False,
    "description":"""
Eine einfache lineare Regression mit Gelenken
"""
}
