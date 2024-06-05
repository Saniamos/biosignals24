import torch
from torch import nn
from datetime import datetime

"""
Selbst designtes CNN
Nutzt Datensatz mit Knochen und Rotation
Daten nicht zufällig.
"""

name = "OwnBaselineCNN"


class NeuralNetwork(nn.Module):
    def __init__(self, output_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),  # 282*160
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),  # 94*53
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 47*26
            nn.Flatten(),
            nn.Linear(47*26*32, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


hyper_params_default = {
    "name": f"{name}_default",
    "epochs": 50,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [100],
    "lr_decay_gamma": 1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": True,
    "output_size": 126,
    "shuffle": True,
    "batch_size": 64,
    "normalize": False,
}

hyper_params_no_rot = {
    "name": f"{name}_no_rot",
    "epochs": 50,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [100],
    "lr_decay_gamma": 1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": False,
    "output_size": 63,
    "shuffle": True,
    "batch_size": 64,
    "normalize": False,
}

hyper_params_low_lr_32_batch_size_normalized = {
    "name": f"{name}_low_lr_32_batch_size_normalized",
    "epochs": 50,
    "learning_rate": 1e-5,
    "lr_decay_milestones": [100],
    "lr_decay_gamma": 1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "use_rotation_data": True,
    "output_size": 126,
    "shuffle": True,
    "batch_size": 32,
    "normalize": True,
}
