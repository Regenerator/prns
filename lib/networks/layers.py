import numpy as np

import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, mean, std, mode='3d'):
        super(Scale, self).__init__()
        if mode == '3d':
            self.register_buffer('mean', torch.from_numpy(np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1, 1)))
            self.register_buffer('std', torch.from_numpy(np.array(std, dtype=np.float32).reshape(1, -1, 1, 1, 1)))
        elif mode == '2d':
            self.register_buffer('mean', torch.from_numpy(np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)))
            self.register_buffer('std', torch.from_numpy(np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)))
        elif mode == '1d':
            self.register_buffer('mean', torch.from_numpy(np.array(mean, dtype=np.float32).reshape(1, -1, 1)))
            self.register_buffer('std', torch.from_numpy(np.array(std, dtype=np.float32).reshape(1, -1, 1)))
        elif mode == '0d':
            self.register_buffer('mean', torch.from_numpy(np.array(mean, dtype=np.float32).reshape(1, -1)))
            self.register_buffer('std', torch.from_numpy(np.array(std, dtype=np.float32).reshape(1, -1)))

    def forward(self, x):
        return (x - self.mean) / self.std


class AddNoise(nn.Module):
    def __init__(self, scale):
        super(AddNoise, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x + self.scale * torch.randn_like(x)


class AffineTransformation(nn.Module):
    def __init__(self, in_features):
        super(AffineTransformation, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(1, in_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, in_features, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(std=1.0)
        nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input):
        return self.weight * input + self.bias


class ExpTransformation(nn.Module):
    def __init__(self, in_features):
        super(ExpTransformation, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(1, in_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, in_features, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input):
        return torch.exp(self.weight) * input + self.bias
