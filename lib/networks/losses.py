import numpy as np

import torch
import torch.nn as nn


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu1, logvar1, mu2=None, logvar2=None):
        if mu2 is None:
            if mu1 is None:
                loss = torch.zeros(1).cuda()
            else:
                loss = -0.5 * torch.mean(torch.sum(1 + logvar1 - (logvar1.exp() + mu1.pow(2)), dim=1))
        else:
            if mu1 is None:
                loss = -0.5 * torch.mean(torch.sum(1 - logvar2 - ((1 + mu2.pow(2)) / logvar2.exp()), dim=1))
            else:
                loss = -0.5 * torch.mean(
                    torch.sum(1 + logvar1 - logvar2 - ((logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp()), dim=1)
                )
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing_mode = kwargs.get('label_smoothing_mode')
        self.label_smoothing_rate = kwargs.get('label_smoothing_rate')

        if self.label_smoothing_mode == 'None':
            self.shift = 0.
        else:
            self.shift = torch.from_numpy(np.array([self.label_smoothing_rate / (1. - 2. * self.label_smoothing_rate)],
                                                   dtype=np.float32)).cuda()

    def forward(self, logprobs, targets):
        if self.shift > 0.:
            if self.label_smoothing_mode == 'Const':
                shifted_targets = torch.add(targets, self.shift)
            elif self.label_smoothing_mode == 'Random':
                shifted_targets = torch.add(targets, torch.abs(torch.randn_like(targets).mul(self.shift)))
            shifted_targets = shifted_targets / shifted_targets.sum(dim=1, keepdim=True)
        else:
            shifted_targets = targets

        return -torch.sum(shifted_targets * logprobs) / logprobs.size(0)


class MeanSquaredL2Norm(nn.Module):
    def __init__(self):
        super(MeanSquaredL2Norm, self).__init__()

    def forward(self, input):
        loss = torch.mean(torch.sum(input**2, 1))
        return loss


class GNetLoss(nn.Module):
    def __init__(self, **kwargs):
        super(GNetLoss, self).__init__()
        self.CEL = CrossEntropyLoss(**kwargs)

    def forward(self, inputs, targets):
        CEL = self.CEL(inputs['logprobs'], targets)
        return CEL


class TLNLoss(nn.Module):
    def __init__(self, **kwargs):
        super(TLNLoss, self).__init__()
        self.kl_weight = kwargs.get('kl_weight')
        self.CEL = CrossEntropyLoss(**kwargs)
        self.L2 = MeanSquaredL2Norm()

    def forward(self, inputs, targets):
        CEL = self.CEL(inputs['logprobs'], targets)
        if inputs['img_prior_mus'] is None:
            CL2 = self.L2(inputs['vox_posterior_mus'] - inputs['vox_prior_mus'])
        else:
            CL2 = self.L2(inputs['vox_posterior_mus'] - inputs['img_prior_mus'])
        return CEL + self.kl_weight * CL2, CEL, CL2


class CVAELoss(nn.Module):
    def __init__(self, **kwargs):
        super(CVAELoss, self).__init__()
        self.kl_weight = kwargs.get('kl_weight')
        self.CEL = CrossEntropyLoss(**kwargs)
        self.KLD = KLDivergence()

    def forward(self, inputs, targets):
        CEL = self.CEL(inputs['logprobs'], targets)
        if inputs['img_prior_mus'] is None:
            KLDI = self.KLD(inputs['vox_posterior_mus'], inputs['vox_posterior_logvars'],
                            mu2=inputs['vox_prior_mus'], logvar2=inputs['vox_prior_logvars'])
        else:
            KLDI = self.KLD(inputs['vox_posterior_mus'], inputs['vox_posterior_logvars'],
                            mu2=inputs['img_prior_mus'], logvar2=inputs['img_prior_logvars'])
        return CEL + self.kl_weight * KLDI, CEL, KLDI


class DVAELoss(nn.Module):
    def __init__(self, **kwargs):
        super(DVAELoss, self).__init__()
        self.kl_weight = kwargs.get('kl_weight')
        self.kl_ratio = kwargs.get('kl_ratio')

        self.CEL = CrossEntropyLoss(**kwargs)
        self.KLD = KLDivergence()

    def forward(self, inputs, targets):
        CEL = self.CEL(inputs['logprobs'], targets)
        KLDV = self.KLD(inputs['vox_posterior_mus'], inputs['vox_posterior_logvars'],
                        mu2=inputs['vox_prior_mus'], logvar2=inputs['vox_prior_logvars'])
        KLDI = self.KLD(inputs['vox_posterior_mus'], inputs['vox_posterior_logvars'],
                        mu2=inputs['img_prior_mus'], logvar2=inputs['img_prior_logvars'])
        return CEL + self.kl_weight * (self.kl_ratio * KLDV + (1.0 - self.kl_ratio) * KLDI), CEL, KLDV, KLDI
