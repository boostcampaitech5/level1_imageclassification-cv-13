"""
models.py
"""

import torch
from torch import nn


def init_model_dict():
    model_dict = {
        'model': 'Model',
    }

    return model_dict


class Model(nn.Module):
    def __init__(self, n_class):
        super(Model, self).__init__()
        self.n_class = n_class

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv2d(16, n_class, kernel_size=3, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(self.conv_2(x))
        x = self.avg_pool(x)
        x = self.relu(x)
        x = x.view(-1, self.n_class)
        return x
    
