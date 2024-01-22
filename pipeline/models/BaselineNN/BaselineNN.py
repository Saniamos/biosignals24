import torch
from torch import nn
from datetime import datetime

name = "BaselineNN"


class BaselineNN(nn.Module):
    def __init__(self, output_size):
        super(BaselineNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(480*848, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    
hyper_params_default = {
    "name": f"{name}_default",
    "epochs": 100,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [0],
    "lr_decay_gamma": 1.0,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": True,
    "output_size": 126,
    "shuffle": True,
    "batch_size": 128,
    "normalize": False,
}
    
hyper_params_no_rot = {
    "name": f"{name}_no_rot",
    "epochs": 100,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [0],
    "lr_decay_gamma": 1.0,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": False,
    "output_size": 63,
    "shuffle": True,
    "batch_size": 128,
    "normalize": False,
}
    
hyper_params_joints = {
    "name": f"{name}_joints",
    "epochs": 100,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [0],
    "lr_decay_gamma": 1.0,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": True,
    "output_size": 60,
    "shuffle": True,
    "batch_size": 128,
    "normalize": False,
}