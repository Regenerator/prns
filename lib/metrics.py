import numpy as np

from torchvision.transforms import Compose


class Accuracy(object):
    def __init__(self, keep_grids=True):
        self.keep_grids = keep_grids

    def __call__(self, sample):
        sample['accuracy'] = 100. * (sample['reco_grid'] == sample['true_grid']).sum() / sample['true_grid'].shape[0]**3
        if not self.keep_grids:
            del sample['reco_grid']
            del sample['true_grid']
        return sample


class IoU(object):
    def __init__(self, keep_grids=True):
        self.keep_grids = keep_grids

    def __call__(self, sample):
        sample['iou'] = 100. * np.logical_and(sample['reco_grid'], sample['true_grid']).sum() / \
            np.logical_or(sample['reco_grid'], sample['true_grid']).sum()
        if not self.keep_grids:
            del sample['reco_grid']
            del sample['true_grid']
        return sample


def ComposeMetrics(**kwargs):
    metrics = [Accuracy(keep_grids=True), IoU(keep_grids=True)]
    if kwargs.get('accuracy', True):
        metrics.append(Accuracy())
    if kwargs.get('iou', True):
        metrics.append(IoU())

    if len(metrics) > 0:
        # metrics[-1].keep_grids = False
        return Compose(metrics)
    else:
        return None
