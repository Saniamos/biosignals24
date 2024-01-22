import torch
from torch import nn
from datetime import datetime

"""
Genauso wie CNNModel_LiChan2014 nur mit AveragePooling Layer

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network.
Average Pooling als ersten Layern hinzugefügt.
"""

name = "CNN_Pelvis_Location"

class NeuralNetwork(nn.Module):
    def __init__(self, hyper_params):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(  # 848 x 480
            nn.AvgPool2d(kernel_size=3, stride=3),  # 282*160
            nn.Conv2d(1, 32, hyper_params["kernel_size"], hyper_params["stride"], hyper_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 141*80
            nn.Conv2d(32, 32, hyper_params["kernel_size"], hyper_params["stride"], hyper_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 70*40
            nn.Conv2d(32, 64, hyper_params["kernel_size"], hyper_params["stride"], hyper_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 35*20
            nn.LocalResponseNorm(32, hyper_params["local_response_norm_alpha"], hyper_params["local_response_norm_beta"]),
            nn.Conv2d(64, 64, hyper_params["kernel_size"], hyper_params["stride"], hyper_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 17*10
            nn.Flatten(),
            nn.Linear(17*10*64, 1024),
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
    "number_joints": 1,
    "output_size": 3,
    "local_response_norm_alpha": 0.0025,
    "local_response_norm_beta": 0.75,
    "use_rotation_data": False,
    "shuffle": True,
    "batch_size": 64,
    "normalize": False,
    "description": ""
}
