import cv2
import numpy as np

from torchvision.transforms import Compose


class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, images):
        numpy_images = []
        for image in images:
            numpy_images.append(np.transpose(np.array(image, dtype=np.float32) / 255., (2, 0, 1)))
        return np.array(numpy_images)


class Resize(object):
    def __init__(self, **kwargs):
        self.size = kwargs.get('image_size')

    def __call__(self, images):
        return np.transpose(cv2.resize(
            np.transpose(images.reshape(-1, images.shape[2], images.shape[3]), (1, 2, 0)),
            (self.size[0], self.size[1])
        ), (2, 0, 1)).reshape(images.shape[0], images.shape[1], self.size[0], self.size[1])


class Pad(object):
    def __init__(self, **kwargs):
        self.pad_size = kwargs.get('image_pad_size')

    def __call__(self, images):
        padded = np.zeros((images.shape[0], images.shape[1],
                           images.shape[2] + 2 * self.pad_size[0],
                           images.shape[3] + 2 * self.pad_size[1]), dtype=np.float32)
        padded[:, :, self.pad_size[0]:-self.pad_size[0], self.pad_size[1]:-self.pad_size[1]] = images
        return padded


class AddGrayscale(object):
    def __init__(self):
        self.r = 0.299
        self.g = 0.587
        self.b = 0.114

    def __call__(self, images):
        return np.hstack((
            np.expand_dims(self.r * images[:, 0] + self.g * images[:, 1] + self.b * images[:, 2], 1),
            images
        ))


class NormalizeImages(object):
    def __init__(self, **kwargs):
        self.mean = np.array(kwargs.get('image_means'), dtype=np.float32)
        self.std = np.array(kwargs.get('image_stds'), dtype=np.float32)

    def __call__(self, images):
        return (images - self.mean.reshape(1, -1, 1, 1)) / self.std.reshape(1, -1, 1, 1)


class AddNoise2Images(object):
    def __init__(self, **kwargs):
        self.scale = kwargs.get('image_noise_scale')

    def __call__(self, images):
        return np.clip(images + np.float32(np.random.normal(scale=self.scale, size=images.shape)), 0.0, 1.0)


class RemoveAlpha(object):
    def __init__(self):
        pass

    def __call__(self, images):
        return images[:, :4]


def ComposeImageTransformation(**kwargs):
    image_transformations = []
    image_transformations.append(ToNumpy())
    if kwargs.get('image_resize'):
        image_transformations.append(Resize(**kwargs))
    if kwargs.get('image_pad'):
        image_transformations.append(Pad(**kwargs))
    if kwargs.get('image_add_grayscale'):
        image_transformations.append(AddGrayscale())
    if kwargs.get('image_normalize'):
        image_transformations.append(NormalizeImages(**kwargs))
    if kwargs.get('image_noise'):
        image_transformations.append(AddNoise2Images(**kwargs))
    if kwargs.get('image_remove_alpha'):
        image_transformations.append(RemoveAlpha())

    if len(image_transformations) == 0:
        return None
    else:
        return Compose(image_transformations)


class AddNoise2Voxels(object):
    def __init__(self, **kwargs):
        self.scale = kwargs.get('voxel_noise_scale')

    def __call__(self, voxels):
        return np.clip(voxels + np.float32(np.random.normal(scale=self.scale, size=voxels.shape)), 0.0, 1.0)


class NormalizeVoxels(object):
    def __init__(self, **kwargs):
        self.mean = np.array(kwargs.get('voxel_means'), dtype=np.float32)
        self.std = np.array(kwargs.get('voxel_stds'), dtype=np.float32)

    def __call__(self, grid):
        return (grid - self.mean.reshape(-1, 1, 1, 1)) / self.std.reshape(-1, 1, 1, 1)


def ComposeGridTransformation(**kwargs):
    grid_transformations = []
    if kwargs.get('voxel_noise'):
        grid_transformations.append(AddNoise2Voxels(**kwargs))
    if kwargs.get('voxel_normalize'):
        grid_transformations.append(NormalizeVoxels(**kwargs))

    if len(grid_transformations) == 0:
        return None
    else:
        return Compose(grid_transformations)
