from collections import OrderedDict

import torch
import torch.nn as nn

from .layers import ExpTransformation


class FeatureEncoder(nn.Module):
    def __init__(self, n_layers, in_features, latent_space_size, deterministic=False):
        super(FeatureEncoder, self).__init__()
        self.n_layers = n_layers
        self.in_features = in_features
        self.latent_space_size = latent_space_size
        self.deterministic = deterministic

        if n_layers > 0:
            self.features = nn.Sequential()
            for i in range(n_layers):
                self.features.add_module('mlp{}'.format(i), nn.Linear(in_features, in_features, bias=False))
                self.features.add_module('mlp{}_bn'.format(i), nn.BatchNorm1d(in_features))
                self.features.add_module('mlp{}_relu'.format(i), nn.ReLU(inplace=True))

        self.mus = nn.Sequential(OrderedDict([
            ('mu_mlp0', nn.Linear(in_features, latent_space_size, bias=True))
        ]))
        with torch.no_grad():
            self.mus[-1].weight.data.normal_(std=0.033)
            nn.init.constant_(self.mus[-1].bias.data, 0.0)

        if not self.deterministic:
            self.logvars = nn.Sequential(OrderedDict([
                ('logvar_mlp0', nn.Linear(in_features, latent_space_size, bias=True))
            ]))
            with torch.no_grad():
                self.logvars[-1].weight.data.normal_(std=0.033)
                nn.init.constant_(self.logvars[-1].bias.data, 0.0)

    def forward(self, input):
        if self.n_layers > 0:
            features = self.features(input)
        else:
            features = input

        if self.deterministic:
            return self.mus(features)
        else:
            return self.mus(features), self.logvars(features)


class AllCNNImageEncoder(nn.Module):
    def __init__(self, init_channels, init_features,
                 num_features, bottleneck_features,
                 latent_space_encoding=True,
                 enc_conditioning=False, dec_conditioning=False,
                 condition_features=None):
        super(AllCNNImageEncoder, self).__init__()
        self.init_channels = init_channels
        self.init_features = init_features
        self.num_features = num_features
        self.bottleneck_features = bottleneck_features
        self.latent_space_encoding = latent_space_encoding
        self.enc_conditioning = enc_conditioning
        self.dec_conditioning = dec_conditioning
        self.condition_features = condition_features

        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(init_channels, init_features, 10, stride=1, padding=0, bias=False)),
            ('conv0_bn', nn.BatchNorm2d(init_features)),
            ('conv0_relu', nn.ReLU(inplace=True))
        ])))

        self.sconvs = nn.ModuleList()
        self.sconvs.append(nn.Sequential(OrderedDict([
            ('sconv0', nn.Conv2d(init_features, num_features[0], 4, stride=2, padding=1, bias=False)),
            ('sconv0_bn', nn.BatchNorm2d(num_features[0])),
            ('sconv0_relu', nn.ReLU(inplace=True))
        ])))

        if self.enc_conditioning or self.dec_conditioning:
            self.conds = nn.ModuleList()
        if self.enc_conditioning:
            self.enc_w = nn.ModuleList()
            self.enc_b = nn.ModuleList()
        if self.dec_conditioning:
            self.dec_w = nn.ModuleList()
            self.dec_b = nn.ModuleList()

        for i in range(1, len(num_features)):
            self.convs.append(nn.Sequential(OrderedDict([
                ('conv{}'.format(i), nn.Conv2d(num_features[i - 1], num_features[i - 1], 3, padding=1, bias=False)),
                ('conv{}_bn'.format(i), nn.BatchNorm2d(num_features[i - 1])),
                ('conv{}_relu'.format(i), nn.ReLU(inplace=True))
            ])))

            if self.enc_conditioning or self.dec_conditioning:
                self.conds.append(nn.Sequential(OrderedDict([
                    ('cond{}_0'.format(i - 1), nn.Linear(num_features[i - 1], condition_features[i - 1], bias=False)),
                    ('cond{}_0_bn'.format(i - 1), nn.BatchNorm1d(condition_features[i - 1])),
                    ('cond{}_0_relu'.format(i - 1), nn.ReLU(inplace=True))
                ])))

            if self.enc_conditioning:
                self.enc_w.append(nn.Sequential(OrderedDict([
                    ('enc_w{}_0'.format(i - 1), nn.Linear(condition_features[i - 1],
                                                          condition_features[i - 1],
                                                          bias=True))
                ])))
                with torch.no_grad():
                    self.enc_w[i - 1][-1].weight.data.normal_(std=0.033)
                    nn.init.constant_(self.enc_w[i - 1][-1].bias.data, 0.0)

                self.enc_b.append(nn.Sequential(OrderedDict([
                    ('enc_b{}_0'.format(i - 1), nn.Linear(condition_features[i - 1],
                                                          condition_features[i - 1],
                                                          bias=True))
                ])))
                with torch.no_grad():
                    self.enc_b[i - 1][-1].weight.data.normal_(std=0.033)
                    nn.init.constant_(self.enc_b[i - 1][-1].bias.data, 0.0)

            if self.dec_conditioning:
                self.dec_w.append(nn.Sequential(OrderedDict([
                    ('dec_w{}_0'.format(i - 1), nn.Linear(condition_features[i - 1],
                                                          condition_features[i - 1],
                                                          bias=True))
                ])))
                with torch.no_grad():
                    self.dec_w[i - 1][-1].weight.data.normal_(std=0.033)
                    nn.init.constant_(self.dec_w[i - 1][-1].bias.data, 0.0)

                self.dec_b.append(nn.Sequential(OrderedDict([
                    ('dec_b{}_0'.format(i - 1), nn.Linear(condition_features[i - 1],
                                                          condition_features[i - 1],
                                                          bias=True))
                ])))
                with torch.no_grad():
                    self.dec_b[i - 1][-1].weight.data.normal_(std=0.033)
                    nn.init.constant_(self.dec_b[i - 1][-1].bias.data, 0.0)

            self.sconvs.append(nn.Sequential(OrderedDict([
                ('sconv{}'.format(i), nn.Conv2d(num_features[i - 1], num_features[i], 4,
                                                stride=2, padding=1, bias=False)),
                ('sconv{}_bn'.format(i), nn.BatchNorm2d(num_features[i])),
                ('sconv{}_relu'.format(i), nn.ReLU(inplace=True))
            ])))

        if self.latent_space_encoding:
            self.bottleneck = nn.Sequential(OrderedDict([
                ('bottleneck', nn.Linear(4 * num_features[-1], bottleneck_features, bias=False)),
                ('bottleneck_bn', nn.BatchNorm1d(bottleneck_features)),
                ('bottleneck_relu', nn.ReLU(inplace=True))
            ]))

    def forward(self, input):
        if self.enc_conditioning:
            enc_w = []
            enc_b = []

        if self.dec_conditioning:
            dec_w = []
            dec_b = []

        features = input
        for i in range(len(self.convs)):
            features = self.convs[i](features)

            if i > 0:
                if self.enc_conditioning or self.dec_conditioning:
                    cond = self.conds[i - 1](nn.functional.avg_pool2d(
                        features, (features.shape[2], features.shape[3]), 1, 0
                    ).squeeze(3).squeeze(2))

                if self.enc_conditioning:
                    enc_w.append(torch.exp(self.enc_w[i - 1](cond)))
                    enc_b.append(self.enc_b[i - 1](cond))

                if self.dec_conditioning:
                    dec_w.append(torch.exp(self.dec_w[i - 1](cond)))
                    dec_b.append(self.dec_b[i - 1](cond))

            features = self.sconvs[i](features)

        output = {}
        if self.latent_space_encoding:
            output['encoded'] = self.bottleneck(features.view(features.shape[0], -1))
        if self.enc_conditioning:
            output['enc_w'], output['enc_b'] = enc_w, enc_b
        if self.dec_conditioning:
            output['dec_w'], output['dec_b'] = dec_w, dec_b
        return output


class ResNetVoxelGridEncoder(nn.Module):
    def __init__(self, init_channels, init_features,
                 num_features, bottleneck_features, conditioning):
        super(ResNetVoxelGridEncoder, self).__init__()
        self.init_channels = init_channels
        self.init_features = init_features
        self.num_features = num_features
        self.bottleneck_features = bottleneck_features
        self.conditioning = conditioning

        self.init_conv = nn.Sequential(OrderedDict([
            ('init_conv', nn.Conv3d(init_channels, init_features, 3, padding=1, bias=False)),
            ('init_conv_bn', nn.BatchNorm3d(init_features, affine=(not self.conditioning)))
        ]))
        if self.conditioning:
            self.init_at = ExpTransformation(init_features)
        self.init_relu = nn.ReLU(inplace=True)

        self.convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        if self.conditioning:
            self.ats = nn.ModuleList()
        self.poolings = nn.ModuleList()
        for i in range(len(num_features)):
            cur_features = init_features if i == 0 else num_features[i - 1]

            self.skip_convs.append(nn.Sequential(OrderedDict([
                ('conv{}_skip'.format(i), nn.Conv3d(cur_features, num_features[i] // 2, 1, padding=0, bias=False))
            ])))

            self.convs.append(nn.Sequential(OrderedDict([
                ('conv{}_0'.format(i), nn.Conv3d(cur_features, num_features[i] // 2, 3, padding=1, bias=False)),
                ('conv{}_0_bn'.format(i), nn.BatchNorm3d(num_features[i] // 2)),
                ('conv{}_0_relu'.format(i), nn.ReLU(inplace=True)),
                ('conv{}_1'.format(i), nn.Conv3d(num_features[i] // 2, num_features[i] // 2, 3, padding=1, bias=False))
            ])))
            if i == 0:
                self.convs[i].add_module('conv{}_1_bn'.format(i), nn.BatchNorm3d(num_features[i] // 2))
                self.convs[i].add_module('conv{}_1_relu'.format(i), nn.ReLU(inplace=True))
                self.convs[i].add_module('conv{}_2'.format(i), nn.Conv3d(num_features[i] // 2, num_features[i] // 2,
                                                                         3, padding=1, bias=False))

            self.bns.append(nn.Sequential(OrderedDict([
                ('conv{}_bn'.format(i), nn.BatchNorm3d(num_features[i], affine=(not self.conditioning)))
            ])))

            if self.conditioning:
                self.ats.append(nn.Sequential(OrderedDict([
                    ('conv{}_at'.format(i), ExpTransformation(num_features[i]))
                ])))

            self.poolings.append(nn.Sequential(OrderedDict([
                ('conv{}_relu'.format(i), nn.ReLU(inplace=True)),
                ('conv{}_ap'.format(i), nn.AvgPool3d(2, stride=2, padding=0))
            ])))

        self.bottleneck = nn.Sequential(OrderedDict([
            ('bottleneck', nn.Linear(8 * num_features[-1], bottleneck_features, bias=False)),
            ('bottleneck_bn', nn.BatchNorm1d(bottleneck_features)),
            ('bottleneck_relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, grid, cond_w=None, cond_b=None):
        features = self.init_conv(grid)
        if self.conditioning:
            if cond_w is None:
                features = self.init_at(features)
            else:
                features = cond_w[0].unsqueeze(2).unsqueeze(3).unsqueeze(4) * features + \
                    cond_b[0].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        features = self.init_relu(features)
        for i in range(len(self.convs)):
            features = self.bns[i](torch.cat([self.convs[i](features), self.skip_convs[i](features)], dim=1))
            if self.conditioning:
                if cond_w is None:
                    features = self.ats[i](features)
                else:
                    features = cond_w[i + 1].unsqueeze(2).unsqueeze(3).unsqueeze(4) * features + \
                        cond_b[i + 1].unsqueeze(2).unsqueeze(3).unsqueeze(4)
            features = self.poolings[i](features)
        features = self.bottleneck(features.view(features.shape[0], -1))
        return features
