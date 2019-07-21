import torch
import torch.nn as nn

from .layers import Scale, AddNoise
from .encoders import AllCNNImageEncoder
from .encoders import ResNetVoxelGridEncoder
from .encoders import FeatureEncoder
from .decoders import LatentSpaceDecoder
from .decoders import ResNetVoxelGridDecoder


class GNet(nn.Module):
    def __init__(self, **kwargs):
        super(GNet, self).__init__()

        self.img_n_views = kwargs.get('img_n_views')
        self.img_init_channels = kwargs.get('img_init_channels')
        self.img_init_features = kwargs.get('img_init_features')
        self.img_num_features = kwargs.get('img_num_features')
        self.img_latent_space_encoding = kwargs.get('img_latent_space_encoding')
        self.img_latent_space_n_layers = kwargs.get('img_latent_space_n_layers')
        self.img_bottleneck_features = kwargs.get('img_bottleneck_features')

        self.deterministic = kwargs.get('deterministic')
        self.latent_space_size = kwargs.get('latent_space_size')

        self.dec_bottleneck_features = kwargs.get('dec_bottleneck_features')
        self.dec_num_features = kwargs.get('dec_num_features')
        self.dec_final_features = kwargs.get('dec_final_features')
        self.dec_final_channels = kwargs.get('dec_final_channels')
        self.dec_conditioning = kwargs.get('dec_conditioning')
        self.dec_latent_space_n_layers = kwargs.get('dec_latent_space_n_layers')
        self.conditioning_features = [self.dec_final_features] + self.dec_num_features[::-1]

        self.prediction_mode = kwargs.get('prediction_mode')

        self.img_encoder = AllCNNImageEncoder(self.img_init_channels, self.img_init_features,
                                              self.img_num_features, self.img_bottleneck_features,
                                              self.img_latent_space_encoding,
                                              False, self.dec_conditioning,
                                              self.conditioning_features)

        if self.img_latent_space_encoding:
            self.img_feature_encoder = FeatureEncoder(self.img_latent_space_n_layers, self.img_bottleneck_features,
                                                      self.latent_space_size, deterministic=self.deterministic)
        else:
            self.mus = nn.Parameter(torch.Tensor(1, self.latent_space_size))
            with torch.no_grad():
                self.mus.data.normal_(std=0.33)
            if not self.deterministic:
                self.logvars = nn.Parameter(torch.Tensor(1, self.latent_space_size))
                with torch.no_grad():
                    self.logvars.data.normal_(std=0.33)

        self.vox_latent_space_decoder = LatentSpaceDecoder(self.dec_latent_space_n_layers, self.latent_space_size,
                                                           self.dec_bottleneck_features)

        self.vox_grid_decoder = ResNetVoxelGridDecoder(self.dec_bottleneck_features, self.dec_num_features,
                                                       self.dec_final_features, self.dec_final_channels,
                                                       self.dec_conditioning)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, images, voxels):
        output = {}

        encoded = self.img_encoder(images.view(images.shape[0] * images.shape[1],
                                               images.shape[2], images.shape[3], images.shape[4]))
        if self.img_latent_space_encoding:
            img_mus = torch.max(encoded['encoded'].view(images.shape[0], images.shape[1],
                                                        self.img_bottleneck_features), dim=1)[0]
        else:
            img_mus = None

        if self.dec_conditioning:
            dec_ws = encoded['dec_w']
            dec_bs = encoded['dec_b']
            for i in range(len(encoded['dec_w'])):
                dec_ws[i] = torch.max(dec_ws[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
                dec_bs[i] = torch.max(dec_bs[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
        else:
            dec_ws = dec_bs = None

        if self.deterministic:
            if self.img_latent_space_encoding:
                img_mus = self.img_feature_encoder(img_mus)
            else:
                img_mus = self.mus.expand(images.shape[0], self.latent_space_size)
            output['logprobs'] = self.vox_grid_decoder(
                self.vox_latent_space_decoder(img_mus),
                cond_w=dec_ws, cond_b=dec_bs
            )
        else:
            if self.img_latent_space_encoding:
                img_mus, img_logvars = self.img_feature_encoder(img_mus)
            else:
                img_mus = self.mus.expand(images.shape[0], self.latent_space_size)
                img_logvars = self.logvars.expand(images.shape[0], self.latent_space_size)

            if self.prediction_mode:
                output['logprobs'] = self.vox_grid_decoder(
                    self.vox_latent_space_decoder(img_mus),
                    cond_w=dec_ws, cond_b=dec_bs
                )
            else:
                output['logprobs'] = self.vox_grid_decoder(
                    self.vox_latent_space_decoder(self.reparameterize(img_mus, img_logvars)),
                    cond_w=dec_ws, cond_b=dec_bs
                )

        return output


class CVAE(nn.Module):
    def __init__(self, **kwargs):
        super(CVAE, self).__init__()

        self.img_n_views = kwargs.get('img_n_views')
        self.img_init_channels = kwargs.get('img_init_channels')
        self.img_init_features = kwargs.get('img_init_features')
        self.img_num_features = kwargs.get('img_num_features')
        self.img_latent_space_encoding = kwargs.get('img_latent_space_encoding')
        self.img_latent_space_n_layers = kwargs.get('img_latent_space_n_layers')
        self.img_bottleneck_features = kwargs.get('img_bottleneck_features')

        self.enc_scaler = kwargs.get('voxel_normalize')
        self.enc_noise = kwargs.get('voxel_noise')
        self.enc_init_channels = kwargs.get('enc_init_channels')
        self.enc_init_features = kwargs.get('enc_init_features')
        self.enc_num_features = kwargs.get('enc_num_features')
        self.enc_bottleneck_features = kwargs.get('enc_bottleneck_features')
        self.enc_conditioning = kwargs.get('enc_conditioning')
        self.enc_latent_space_n_layers = kwargs.get('enc_latent_space_n_layers')

        self.latent_space_size = kwargs.get('latent_space_size')
        self.deterministic = kwargs.get('deterministic')

        self.dec_bottleneck_features = kwargs.get('dec_bottleneck_features')
        self.dec_num_features = kwargs.get('dec_num_features')
        self.dec_final_features = kwargs.get('dec_final_features')
        self.dec_final_channels = kwargs.get('dec_final_channels')
        self.dec_conditioning = kwargs.get('dec_conditioning')
        self.dec_latent_space_n_layers = kwargs.get('dec_latent_space_n_layers')
        self.conditioning_features = [self.dec_final_features] + self.dec_num_features[::-1]

        self.prediction_mode = kwargs.get('prediction_mode')

        self.img_encoder = AllCNNImageEncoder(self.img_init_channels, self.img_init_features,
                                              self.img_num_features, self.img_bottleneck_features,
                                              self.img_latent_space_encoding,
                                              self.enc_conditioning, self.dec_conditioning,
                                              self.conditioning_features)

        if self.img_latent_space_encoding:
            self.img_feature_encoder = FeatureEncoder(self.img_latent_space_n_layers, self.img_bottleneck_features,
                                                      self.latent_space_size, deterministic=self.deterministic)

        self.mus = nn.Parameter(torch.Tensor(1, self.latent_space_size))
        with torch.no_grad():
            self.mus.data.normal_(std=0.33)
        if not self.deterministic:
            self.logvars = nn.Parameter(torch.Tensor(1, self.latent_space_size))
            with torch.no_grad():
                self.logvars.data.normal_(std=0.33)

        if self.enc_scaler:
            self.vox_scaler = Scale(kwargs['voxel_means'], kwargs['voxel_stds'], mode='3d')
            if self.enc_noise:
                self.vox_noise = AddNoise(kwargs['voxel_noise_scale'])

        self.vox_grid_encoder = ResNetVoxelGridEncoder(self.enc_init_channels, self.enc_init_features,
                                                       self.enc_num_features, self.enc_bottleneck_features,
                                                       self.enc_conditioning)
        self.vox_feature_encoder = FeatureEncoder(self.enc_latent_space_n_layers, self.enc_bottleneck_features,
                                                  self.latent_space_size, deterministic=self.deterministic)

        self.vox_latent_space_decoder = LatentSpaceDecoder(self.dec_latent_space_n_layers, self.latent_space_size,
                                                           self.dec_bottleneck_features)
        self.vox_grid_decoder = ResNetVoxelGridDecoder(self.dec_bottleneck_features, self.dec_num_features,
                                                       self.dec_final_features, self.dec_final_channels,
                                                       self.dec_conditioning)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, images, voxels):
        output = {}

        encoded = self.img_encoder(images.view(images.shape[0] * images.shape[1],
                                               images.shape[2], images.shape[3], images.shape[4]))

        if self.img_latent_space_encoding:
            output['img_prior_mus'] = torch.max(encoded['encoded'].view(images.shape[0], images.shape[1],
                                                                        self.img_bottleneck_features), dim=1)[0]
        else:
            output['img_prior_mus'] = None

        if self.enc_conditioning:
            enc_ws = encoded['enc_w']
            enc_bs = encoded['enc_b']
            for i in range(len(encoded['enc_w'])):
                enc_ws[i] = torch.max(enc_ws[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
                enc_bs[i] = torch.max(enc_bs[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
        else:
            enc_ws = enc_bs = None

        if self.dec_conditioning:
            dec_ws = encoded['dec_w']
            dec_bs = encoded['dec_b']
            for i in range(len(encoded['dec_w'])):
                dec_ws[i] = torch.max(dec_ws[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
                dec_bs[i] = torch.max(dec_bs[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
        else:
            dec_ws = dec_bs = None

        if self.enc_scaler:
            if self.enc_noise:
                output['vox_posterior_mus'] = self.vox_grid_encoder(
                    self.vox_scaler(voxels), cond_w=enc_ws, cond_b=enc_bs
                )
            else:
                output['vox_posterior_mus'] = self.vox_grid_encoder(
                    self.vox_noise(self.vox_scaler(voxels)), cond_w=enc_ws, cond_b=enc_bs
                )
        else:
            output['vox_posterior_mus'] = self.vox_grid_encoder(voxels, cond_w=enc_ws, cond_b=enc_bs)

        if self.deterministic:
            output['vox_prior_mus'] = self.mus.expand(voxels.shape[0], self.latent_space_size)
            output['vox_posterior_mus'] = self.vox_feature_encoder(output['vox_posterior_mus'])

            if self.img_latent_space_encoding:
                output['img_prior_mus'] = self.img_feature_encoder(output['img_prior_mus'])

            if self.prediction_mode:
                if self.img_latent_space_encoding:
                    output['logprobs'] = self.vox_grid_decoder(
                        self.vox_latent_space_decoder(output['img_prior_mus']),
                        cond_w=dec_ws, cond_b=dec_bs
                    )
                else:
                    output['logprobs'] = self.vox_grid_decoder(
                        self.vox_latent_space_decoder(output['vox_prior_mus']),
                        cond_w=dec_ws, cond_b=dec_bs
                    )
            else:
                output['logprobs'] = self.vox_grid_decoder(
                    self.vox_latent_space_decoder(output['vox_posterior_mus']),
                    cond_w=dec_ws, cond_b=dec_bs
                )

        else:
            output['vox_prior_mus'] = self.mus.expand(voxels.shape[0], self.latent_space_size)
            output['vox_prior_logvars'] = self.logvars.expand(voxels.shape[0], self.latent_space_size)
            output['vox_posterior_mus'], output['vox_posterior_logvars'] = \
                self.vox_feature_encoder(output['vox_posterior_mus'])

            if self.img_latent_space_encoding:
                output['img_prior_mus'], output['img_prior_logvars'] = \
                    self.img_feature_encoder(output['img_prior_mus'])

            if self.prediction_mode:
                if self.img_latent_space_encoding:
                    output['logprobs'] = self.vox_grid_decoder(
                        self.vox_latent_space_decoder(output['img_prior_mus']),
                        cond_w=dec_ws, cond_b=dec_bs
                    )
                else:
                    output['logprobs'] = self.vox_grid_decoder(
                        self.vox_latent_space_decoder(output['vox_prior_mus']),
                        cond_w=dec_ws, cond_b=dec_bs
                    )
            else:
                output['logprobs'] = self.vox_grid_decoder(
                    self.vox_latent_space_decoder(self.reparameterize(output['vox_posterior_mus'],
                                                                      output['vox_posterior_logvars'])),
                    cond_w=dec_ws, cond_b=dec_bs
                )

        return output


class DVAE(nn.Module):
    def __init__(self, **kwargs):
        super(DVAE, self).__init__()

        self.img_n_views = kwargs.get('img_n_views')
        self.img_init_channels = kwargs.get('img_init_channels')
        self.img_init_features = kwargs.get('img_init_features')
        self.img_num_features = kwargs.get('img_num_features')
        self.img_latent_space_encoding = kwargs.get('img_latent_space_encoding')
        self.img_latent_space_n_layers = kwargs.get('img_latent_space_n_layers')
        self.img_bottleneck_features = kwargs.get('img_bottleneck_features')

        self.enc_scaler = kwargs.get('voxel_normalize')
        self.enc_noise = kwargs.get('voxel_noise')
        self.enc_init_channels = kwargs.get('enc_init_channels')
        self.enc_init_features = kwargs.get('enc_init_features')
        self.enc_num_features = kwargs.get('enc_num_features')
        self.enc_bottleneck_features = kwargs.get('enc_bottleneck_features')
        self.enc_conditioning = kwargs.get('enc_conditioning')
        self.enc_latent_space_n_layers = kwargs.get('enc_latent_space_n_layers')

        self.latent_space_size = kwargs.get('latent_space_size')
        self.deterministic = kwargs.get('deterministic')

        self.dec_bottleneck_features = kwargs.get('dec_bottleneck_features')
        self.dec_num_features = kwargs.get('dec_num_features')
        self.dec_final_features = kwargs.get('dec_final_features')
        self.dec_final_channels = kwargs.get('dec_final_channels')
        self.dec_conditioning = kwargs.get('dec_conditioning')
        self.dec_latent_space_n_layers = kwargs.get('dec_latent_space_n_layers')
        self.conditioning_features = [self.dec_final_features] + self.dec_num_features[::-1]

        self.prediction_mode = kwargs.get('prediction_mode')

        self.img_encoder = AllCNNImageEncoder(self.img_init_channels, self.img_init_features,
                                              self.img_num_features, self.img_bottleneck_features,
                                              self.img_latent_space_encoding,
                                              self.enc_conditioning, self.dec_conditioning,
                                              self.conditioning_features)

        if self.img_latent_space_encoding:
            self.img_feature_encoder = FeatureEncoder(self.img_latent_space_n_layers, self.img_bottleneck_features,
                                                      self.latent_space_size, deterministic=self.deterministic)

        self.register_buffer('mus', torch.zeros(1, self.latent_space_size))
        if not self.deterministic:
            self.register_buffer('logvars', torch.zeros(1, self.latent_space_size))

        if self.enc_scaler:
            self.vox_scaler = Scale(kwargs['voxel_means'], kwargs['voxel_stds'], mode='3d')
            if self.enc_noise:
                self.vox_noise = AddNoise(kwargs['voxel_noise_scale'])

        self.vox_grid_encoder = ResNetVoxelGridEncoder(self.enc_init_channels, self.enc_init_features,
                                                       self.enc_num_features, self.enc_bottleneck_features,
                                                       self.enc_conditioning)
        self.vox_feature_encoder = FeatureEncoder(self.enc_latent_space_n_layers, self.enc_bottleneck_features,
                                                  self.latent_space_size, deterministic=self.deterministic)

        self.vox_latent_space_decoder = LatentSpaceDecoder(self.dec_latent_space_n_layers, self.latent_space_size,
                                                           self.dec_bottleneck_features)
        self.vox_grid_decoder = ResNetVoxelGridDecoder(self.dec_bottleneck_features, self.dec_num_features,
                                                       self.dec_final_features, self.dec_final_channels,
                                                       self.dec_conditioning)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, images, voxels):
        output = {}

        encoded = self.img_encoder(images.view(images.shape[0] * images.shape[1],
                                               images.shape[2], images.shape[3], images.shape[4]))

        if self.img_latent_space_encoding:
            output['img_prior_mus'] = torch.max(encoded['encoded'].view(images.shape[0], images.shape[1],
                                                                        self.img_bottleneck_features), dim=1)[0]
        else:
            output['img_prior_mus'] = None

        if self.enc_conditioning:
            enc_ws = encoded['enc_w']
            enc_bs = encoded['enc_b']
            for i in range(len(encoded['enc_w'])):
                enc_ws[i] = torch.max(enc_ws[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
                enc_bs[i] = torch.max(enc_bs[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
        else:
            enc_ws = enc_bs = None

        if self.dec_conditioning:
            dec_ws = encoded['dec_w']
            dec_bs = encoded['dec_b']
            for i in range(len(encoded['dec_w'])):
                dec_ws[i] = torch.max(dec_ws[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
                dec_bs[i] = torch.max(dec_bs[i].view(images.shape[0], images.shape[1],
                                                     self.conditioning_features[i]), dim=1)
        else:
            dec_ws = dec_bs = None

        if self.enc_scaler:
            if self.enc_noise:
                output['vox_posterior_mus'] = self.vox_grid_encoder(
                    self.vox_scaler(voxels), cond_w=enc_ws, cond_b=enc_bs
                )
            else:
                output['vox_posterior_mus'] = self.vox_grid_encoder(
                    self.vox_noise(self.vox_scaler(voxels)), cond_w=enc_ws, cond_b=enc_bs
                )
        else:
            output['vox_posterior_mus'] = self.vox_grid_encoder(voxels, cond_w=enc_ws, cond_b=enc_bs)

        if self.deterministic:
            output['vox_prior_mus'] = self.mus.expand(voxels.shape[0], self.latent_space_size)
            output['vox_posterior_mus'] = self.vox_feature_encoder(output['vox_posterior_mus'])

            if self.img_latent_space_encoding:
                output['img_prior_mus'] = self.img_feature_encoder(output['img_prior_mus'])

            if self.prediction_mode:
                if self.img_latent_space_encoding:
                    output['logprobs'] = self.vox_grid_decoder(
                        self.vox_latent_space_decoder(output['img_prior_mus']),
                        cond_w=dec_ws, cond_b=dec_bs
                    )
                else:
                    output['logprobs'] = self.vox_grid_decoder(
                        self.vox_latent_space_decoder(output['vox_prior_mus']),
                        cond_w=dec_ws, cond_b=dec_bs
                    )
            else:
                output['logprobs'] = self.vox_grid_decoder(
                    self.vox_latent_space_decoder(output['vox_posterior_mus']),
                    cond_w=dec_ws, cond_b=dec_bs
                )

        else:
            output['vox_prior_mus'] = self.mus.expand(voxels.shape[0], self.latent_space_size)
            output['vox_prior_logvars'] = self.logvars.expand(voxels.shape[0], self.latent_space_size)
            output['vox_posterior_mus'], output['vox_posterior_logvars'] = \
                self.vox_feature_encoder(output['vox_posterior_mus'])

            if self.img_latent_space_encoding:
                output['img_prior_mus'], output['img_prior_logvars'] = \
                    self.img_feature_encoder(output['img_prior_mus'])

            if self.prediction_mode:
                if self.img_latent_space_encoding:
                    output['logprobs'] = self.vox_grid_decoder(
                        self.vox_latent_space_decoder(output['img_prior_mus']),
                        cond_w=dec_ws, cond_b=dec_bs
                    )
                else:
                    output['logprobs'] = self.vox_grid_decoder(
                        self.vox_latent_space_decoder(output['vox_prior_mus']),
                        cond_w=dec_ws, cond_b=dec_bs
                    )
            else:
                output['logprobs'] = self.vox_grid_decoder(
                    self.vox_latent_space_decoder(self.reparameterize(output['vox_posterior_mus'],
                                                                      output['vox_posterior_logvars'])),
                    cond_w=dec_ws, cond_b=dec_bs
                )

        return output
