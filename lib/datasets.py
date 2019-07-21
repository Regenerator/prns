import h5py as h5
import numpy as np

from PIL import Image
from time import time
from datetime import datetime
from torch.utils.data import Dataset


class ShapeNetMultiviewDataset(Dataset):
    def __init__(self, path2data, part='train',
                 n_views_used=1, views_shuffle=False,
                 grid_size=32, sample_grids=False,
                 image_transform=None, grid_transform=None):
        super(ShapeNetMultiviewDataset, self).__init__()
        self.path2data = path2data
        self.n_views = 24
        self.n_views_used = n_views_used
        self.views_shuffle = views_shuffle
        self.grid_size = grid_size
        self.sample_grids = sample_grids
        self.image_transform = image_transform
        self.grid_transform = grid_transform

        self.images_file = None
        self.grids_file = None
        self.choose_part(part)
        self.prepare_indices()

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        np.random.seed(datetime.now().second + datetime.now().microsecond)

        if self.images_file is None:
            self.images_file = h5.File('{}ShapeNet_All_images.h5'.format(self.path2data), 'r')
        if self.grids_file is None and self.sample_grids:
            self.grids_file = h5.File(
                '{}ShapeNet_All_grids_full.h5'.format(self.path2data),
                'r'
            )

        start = time()
        sample = {}
        sample['image'] = np.empty((self.n_views_used,) + self.image_size, dtype=np.uint8)
        sample['image'] = []
        img_buf = np.empty(self.image_size, dtype=np.uint8)
        for j in range(self.n_views_used):
            self.images_file[self.part + '_images'].read_direct(img_buf, source_sel=self.img_inds[i][j])
            sample['image'].append(Image.fromarray(np.transpose(img_buf, (1, 2, 0))))
        sample['ilt'] = time() - start

        start = time()
        if self.sample_grids:
            sample['grid'] = np.empty(self.grid_size**3 // 8, dtype=np.uint8)
            self.grids_file[self.part + '_grids'].read_direct(
                sample['grid'],
                source_sel=self.shp_inds[i]
            )
            sample['grid'] = np.float32(np.unpackbits(sample['grid']).reshape(
                1, self.grid_size, self.grid_size, self.grid_size
            ))
            sample['grid'] = np.vstack((np.logical_not(sample['grid']), sample['grid'])).astype(np.float32)
        sample['glt'] = time() - start

        start = time()
        if self.image_transform is not None:
            sample['image'] = self.image_transform(sample['image'])
        sample['itt'] = time() - start

        start = time()
        if self.sample_grids and self.grid_transform is not None:
            sample['grid'] = self.grid_transform(sample['grid'])
        sample['gtt'] = time() - start

        if i == self.size - 1:
            self.prepare_indices()

        return sample

    def choose_part(self, part):
        self.part = part
        with h5.File('{}ShapeNet_All_images.h5'.format(self.path2data), 'r') as fin:
            self.size = fin[part + '_images'].shape[0] // self.n_views_used
            self.image_size = fin[part + '_images'].shape[1:]
        self.close()

    def prepare_indices(self):
        if self.n_views_used == 1:
            self.img_inds = np.arange(self.size).reshape(-1, 1)
            self.shp_inds = np.tile(
                np.arange(self.size // self.n_views).reshape(-1, 1),
                (1, self.n_views)
            ).flatten()
        else:
            if self.views_shuffle:
                self.img_inds = np.random.random(
                    (self.size * self.n_views_used // self.n_views, self.n_views)
                ).argsort(1)
                self.img_inds = self.img_inds + \
                    (np.arange(self.size * self.n_views_used // self.n_views) * self.n_views).reshape(-1, 1)
                self.img_inds = np.sort(self.img_inds.reshape(self.size, self.n_views_used), 1)
            else:
                self.img_inds = np.arange(self.size * self.n_views_used).reshape(self.size, self.n_views_used)
            self.shp_inds = np.tile(
                np.arange(self.size * self.n_views_used // self.n_views).reshape(-1, 1),
                (1, self.n_views // self.n_views_used)
            ).flatten()

    def close(self):
        if self.images_file is not None:
            self.images_file.close()
            self.images_file = None
        if self.grids_file is not None:
            self.grids_file.close()
            self.grids_file = None
