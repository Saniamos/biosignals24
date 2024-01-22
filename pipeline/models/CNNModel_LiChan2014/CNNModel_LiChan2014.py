import torch
from torch import nn
from datetime import datetime

"""
Genauso wie CNNModel_2022_06_22 nur ohne AveragePooling Layer

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network.
"""

name = "CNNModel_LiChan2014"

class NeuralNetwork(nn.Module):
    def __init__(self, hyper_params):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential( # 848 x 480
            nn.Conv2d(1, 32, hyper_params["kernel_size"], hyper_params["stride"], hyper_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 424 x 240
            nn.Conv2d(32, 32, hyper_params["kernel_size"], hyper_params["stride"], hyper_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 212 x 120
            nn.Conv2d(32, 64, hyper_params["kernel_size"], hyper_params["stride"], hyper_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 106 x 60
            nn.LocalResponseNorm(32, hyper_params["local_response_norm_alpha"], hyper_params["local_response_norm_beta"]),
            nn.Conv2d(64, 64, hyper_params["kernel_size"], hyper_params["stride"], hyper_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 53 x 30
            nn.Flatten(),
            nn.Linear(53*30*64, 1024),
            nn.ReLU(),
            nn.Dropout(p=hyper_params["dropout_probability"]),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, hyper_params["output_size"]),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


hyper_params_default = {
    "name": f"{name}_default",
    "epochs": 100,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [100],
    "lr_decay_gamma": 1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "dropout_probability": 0.25,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "number_joints": 21,
    "local_response_norm_alpha": 0.0025,
    "local_response_norm_beta": 0.75,
    "use_rotation_data": True,
    "output_size": 126,
    "shuffle": False,
    "normalize": False,
    "batch_size": 32,
    "description": """
CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network
Nutzt Datensatz mit Knochen und Rotation
Daten nicht zufällig.
""",
}


hyper_params_no_rot = {
    "name": f"{name}_no_rotation",
    "epochs": 100,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [100],
    "lr_decay_gamma": 1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "dropout_probability": 0.25,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "number_joints": 21,
    "local_response_norm_alpha": 0.0025,
    "local_response_norm_beta": 0.75,
    "use_rotation_data": False,
    "output_size": 63,
    "shuffle": False,
    "normalize": False,
    "batch_size": 32,
    "description": """
Genauso wie CNNModel_LiChan2014 nur ohne Rotation als Output des Netzes

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network
Nutzt Datensatz mit Knochen ohne Rotation
Daten nicht zufällig.
"""
}


hyper_params_no_rot_shuffle = {
    "name": f"{name}_no_rotation_shuffle",
    "epochs": 100,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [100],
    "lr_decay_gamma": 1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "dropout_probability": 0.25,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "number_joints": 21,
    "local_response_norm_alpha": 0.0025,
    "local_response_norm_beta": 0.75,
    "use_rotation_data": False,
    "output_size": 63,
    "shuffle": True,
    "normalize": False,
    "batch_size": 32,
    "description": """
Genauso wie CNNModel_LiChan2014 nur ohne Rotation als Output des Netzes

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network
Nutzt Datensatz mit Knochen ohne Rotation
Daten nicht zufällig.
"""
}

hyper_params_no_rot_shuffle_low_lr = {
    "name": f"{name}_no_rotation_shuffle_low_lr",
    "epochs": 100,
    "learning_rate": 1e-5,
    "lr_decay_milestones": [100],
    "lr_decay_gamma": 1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "dropout_probability": 0.25,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "number_joints": 21,
    "local_response_norm_alpha": 0.0025,
    "local_response_norm_beta": 0.75,
    "use_rotation_data": False,
    "output_size": 63,
    "shuffle": True,
    "normalize": False,
    "batch_size": 32,
    "description": """
Genauso wie CNNModel_LiChan2014 nur ohne Rotation als Output des Netzes

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network
Nutzt Datensatz mit Knochen ohne Rotation
Daten nicht zufällig.
"""
}


hyper_params_joints_shuffle = {
    "name": f"{name}_joints_shuffle",
    "epochs": 100,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [20,70],
    "lr_decay_gamma": 0.1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "dropout_probability": 0.25,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "number_joints": 21,
    "local_response_norm_alpha": 0.0025,
    "local_response_norm_beta": 0.75,
    "use_rotation_data": False,
    "output_size": 60,
    "shuffle": True,
    "normalize": False,
    "batch_size": 32,
    "description": """
Genauso wie CNNModel_LiChan2014_26 nur:
 - Gelenke nicht Knochen als Ausabe des Netzes
 - Daten werden dem Netz in zufälliger Reihenfolge gegeben

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network
Nutzt Datensatz mit Knochen ohne Rotation
Daten zufällig.
"""
}


hyper_params_joints_no_shuffle = {
    "name": f"{name}_joints_no_shuffle",
    "epochs": 100,
    "learning_rate": 1e-4,
    "lr_decay_milestones": [20,70],
    "lr_decay_gamma": 0.1,
    "loss_function": nn.MSELoss(),
    "optimizer": torch.optim.Adam,
    "dropout_probability": 0.25,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "number_joints": 21,
    "local_response_norm_alpha": 0.0025,
    "local_response_norm_beta": 0.75,
    "use_rotation_data": False,
    "output_size": 60,
    "shuffle": False,
    "normalization": False,
    "batch_size": 32,
    "description": """
Genauso wie CNNModel_LiChan2014 nur:
 - Gelenke nicht Knochen als Ausabe des Netzes

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network
Nutzt Datensatz mit Knochen ohne Rotation
Daten nicht zufällig.
"""
}

