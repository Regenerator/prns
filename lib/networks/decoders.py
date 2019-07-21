from collections import OrderedDict

import torch
import torch.nn as nn

from .layers import ExpTransformation


class LatentSpaceDecoder(nn.Module):
    def __init__(self, n_layers, latent_space_size, out_features):
        super(LatentSpaceDecoder, self).__init__()

        self.n_layers = n_layers
        self.latent_space_size = latent_space_size
        self.out_features = out_features

        self.features = nn.Sequential()
        for i in range(n_layers + 1):
            cur_features = latent_space_size if i == 0 else out_features
            self.features.add_module('mlp{}'.format(i), nn.Linear(cur_features, out_features, bias=False))
            self.features.add_module('mlp{}_bn'.format(i), nn.BatchNorm1d(out_features))
            self.features.add_module('mlp{}_relu'.format(i), nn.ReLU(inplace=True))

    def forward(self, z):
        return self.features(z)


class ResNetVoxelGridDecoder(nn.Module):
    def __init__(self, bottleneck_features, num_features,
                 final_features, final_channels, conditioning):
        super(ResNetVoxelGridDecoder, self).__init__()
        self.bottleneck_features = bottleneck_features
        self.num_features = num_features
        self.final_features = final_features
        self.final_channels = final_channels
        self.conditioning = conditioning

        self.init_transformations = nn.Sequential(OrderedDict([
            ('bottleneck', nn.Linear(bottleneck_features, 8 * num_features[0], bias=False))
        ]))

        self.bns = nn.ModuleList()
        if self.conditioning:
            self.ats = nn.ModuleList()
        self.unpoolings = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(num_features)):
            new_features = num_features[i + 1] if i < len(num_features) - 1 else final_features

            self.bns.append(nn.Sequential(OrderedDict([
                ('conv{}_bn'.format(i), nn.BatchNorm3d(num_features[i], affine=(not self.conditioning)))
            ])))

            if self.conditioning:
                self.ats.append(nn.Sequential(OrderedDict([
                    ('conv{}_at'.format(i), ExpTransformation(num_features[i]))
                ])))

            self.unpoolings.append(nn.Sequential(OrderedDict([
                ('conv{}_relu'.format(i), nn.ReLU(inplace=True)),
                ('conv{}_up'.format(i), nn.Upsample(scale_factor=2, mode='trilinear'))
            ])))

            self.skip_convs.append(nn.Sequential(OrderedDict([
                ('conv{}_skip'.format(i), nn.Conv3d(num_features[i], new_features // 2, 1, padding=0, bias=False)),
            ])))

            self.convs.append(nn.Sequential(OrderedDict([
                ('conv{}_0'.format(i), nn.Conv3d(num_features[i], new_features // 2, 3, padding=1, bias=False)),
                ('conv{}_0_bn'.format(i), nn.BatchNorm3d(new_features // 2)),
                ('conv{}_0_relu'.format(i), nn.ReLU(inplace=True)),
                ('conv{}_1'.format(i), nn.Conv3d(new_features // 2, new_features // 2, 3, padding=1, bias=False))
            ])))
            if i == len(num_features) - 1:
                self.convs[i].add_module('conv3_1_bn', nn.BatchNorm3d(new_features // 2))
                self.convs[i].add_module('conv3_1_relu', nn.ReLU(inplace=True))
                self.convs[i].add_module('conv3_2'.format(i), nn.Conv3d(
                    new_features // 2, new_features // 2, 3, padding=1, bias=False
                ))

        self.final_bn = nn.Sequential(OrderedDict([
            ('conv3_bn', nn.BatchNorm3d(final_features, affine=(not self.conditioning)))
        ]))

        if self.conditioning:
            self.final_at = nn.Sequential(OrderedDict([
                ('conv3_at', ExpTransformation(final_features))
            ]))

        self.outputs = nn.Sequential(OrderedDict([
            ('conv3_relu', nn.ReLU(inplace=True)),
            ('output', nn.Conv3d(final_features, final_channels, 3, padding=1, bias=True)),
            ('output_logprobs', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, z, cond_w=None, cond_b=None):
        grid = self.init_transformations(z)
        grid = grid.view(grid.shape[0], grid.shape[1] // 8, 2, 2, 2)
        for i in range(len(self.convs)):
            grid = self.bns[i](grid)
            if self.conditioning:
                if cond_w is None:
                    grid = self.ats[i](grid)
                else:
                    grid = cond_w[-(i + 1)].unsqueeze(2).unsqueeze(3).unsqueeze(4) * grid + \
                        cond_b[-(i + 1)].unsqueeze(2).unsqueeze(3).unsqueeze(4)
            grid = self.unpoolings[i](grid)
            grid = torch.cat([self.convs[i](grid), self.skip_convs[i](grid)], dim=1)
        grid = self.final_bn(grid)
        if self.conditioning:
            if cond_w is None:
                grid = self.final_at(grid)
            else:
                grid = cond_w[0].unsqueeze(2).unsqueeze(3).unsqueeze(4) * grid + \
                    cond_b[0].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        grid = self.outputs(grid)
        return grid
