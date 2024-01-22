import torch
from torch import nn
import torchvision

import loss.CompositionalLoss

"""
ResNet50 wie beschrieben im Paper: Compositional Human Pose Regression
"""

name = "ResNet50_SunShangetAl"

def create_nn(hyper_params: dict):
    neural_network = torchvision.models.resnet50(pretrained=False)
    neural_network.fc = nn.Linear(2048, hyper_params['output_size'])
    neural_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return neural_network

hyper_params_l1_loss = {
    "name": f"{name}_l1_loss",
    "epochs": 25,
    "learning_rate": 0.03,
    "lr_decay_milestones": [10, 20],
    "lr_decay_gamma": 0.1,
    "weight_decay": 0.0002,
    "loss_function": nn.L1Loss(),
    "momentum": 0.9,
    "optimizer": torch.optim.SGD,
    "output_size": 126,
    "use_rotation_data": True,
    "shuffle": True,
    "batch_size": 8,
    "normalize": False,
    "description":"""
ResNet50 wie beschrieben im Paper: Compositional Human Pose Regression.
Als Loss kommt L1 zum Einsatz
"""
}

hyper_params_l1_loss_no_rot = {
    "name": f"{name}_l1_loss_no_rot",
    "epochs": 50,
    "learning_rate": 0.03,
    "lr_decay_milestones": [10, 20],
    "lr_decay_gamma": 0.1,
    "weight_decay": 0.0002,
    "loss_function": nn.L1Loss(),
    "momentum": 0.9,
    "optimizer": torch.optim.Adam,
    "output_size": 63,
    "use_rotation_data": False,
    "shuffle": True,
    "batch_size": 8,
    "normalize": False,
    "description":"""
ResNet50 wie beschrieben im Paper: Compositional Human Pose Regression.
Als Loss kommt L1 zum Einsatz
"""
}

hyper_params_mse_loss_no_rot = {
    "name": f"{name}_mse_loss_no_rot",
    "epochs": 50,
    "learning_rate": 0.03,
    "lr_decay_milestones": [10, 20],
    "lr_decay_gamma": 0.1,
    "weight_decay": 0.0002,
    "loss_function": nn.MSELoss(),
    "momentum": 0.9,
    "optimizer": torch.optim.Adam,
    "output_size": 63,
    "use_rotation_data": False,
    "shuffle": True,
    "batch_size": 8,
    "normalize": False,
    "description":"""
ResNet50 wie beschrieben im Paper: Compositional Human Pose Regression.
Als Loss kommt L1 zum Einsatz
"""
}

hyper_params_compositional_loss = {
    "name": f"{name}_compositional_loss",
    "epochs": 25,
    "learning_rate": 0.03,
    "lr_decay_milestones": [10, 20],
    "lr_decay_gamma": 0.1,
    "loss_function": loss.CompositionalLoss,
    "optimizer": torch.optim.SGD,
    "output_size": 63,
    "momentum": 0.9,
    "weight_decay": 0.0002,
    "use_rotation_data": False,
    "shuffle": True,
    "batch_size": 8,
    "normalize": False,
    "description":"""
ResNet50 wie beschrieben im Paper: Compositional Human Pose Regression.
Als Loss kommt der im Paper beschriebene Compositional Loss zum Einsatz.
"""
}

hyper_params_compositional_loss_custom_lr = {
    "name": f"{name}_compositional_loss",
    "epochs": 25,
    "learning_rate": 0.001,
    "lr_decay_milestones": [10, 20],
    "lr_decay_gamma": 0.3,
    "loss_function": loss.CompositionalLoss,
    "optimizer": torch.optim.Adam,
    "output_size": 63,
    "use_rotation_data": False,
    "shuffle": True,
    "batch_size": 8,
    "normalize": True,
    "description":"""
ResNet50 wie beschrieben im Paper: Compositional Human Pose Regression.
Als Loss kommt der im Paper beschriebene Compositional Loss zum Einsatz.
"""
}

