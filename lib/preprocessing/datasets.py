import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class AllImagesDataset(Dataset):
    def __init__(self, path2data, part='train'):
        super(AllImagesDataset, self).__init__()
        self.path2data = path2data
        self.part = part

        self.cats = sorted(os.listdir('{}ShapeNetRendering/'.format(path2data)))
        self.model_names = []
        for cat in self.cats:
            cat_names = sorted([
                name for name in os.listdir('{}ShapeNetRendering/{}'.format(path2data, cat))
                if os.path.isdir('{}ShapeNetRendering/{}/{}'.format(path2data, cat, name))
            ])
            cat_size = len(cat_names)
            if part == 'train':
                cat_names = cat_names[:int(0.8 * cat_size)]
            elif part == 'test':
                cat_names = cat_names[int(0.8 * cat_size):]
            self.model_names += list(map(
                lambda n: '{}ShapeNetRendering/{}/{}'.format(path2data, cat, n),
                cat_names
            ))

    def __len__(self):
        return 24 * len(self.model_names)

    def __getitem__(self, i):
        return np.transpose(np.array(cv2.imread(
            self.model_names[i // 24] + '/rendering/{:02d}.png'.format(i % 24), -1), dtype=np.uint8), (2, 0, 1)
        )


class AllGridDataset(Dataset):
    def __init__(self, path2data, part='train'):
        super(AllGridDataset, self).__init__()
        self.path2data = path2data
        self.part = part

        self.cats = sorted(os.listdir('{}ShapeNetVox32/'.format(path2data)))
        self.model_names = []
        for cat in self.cats:
            cat_names = sorted([
                name for name in os.listdir('{}ShapeNetVox32/{}'.format(path2data, cat))
                if os.path.isdir('{}ShapeNetVox32/{}/{}'.format(path2data, cat, name))
            ])
            cat_size = len(cat_names)
            if part == 'train':
                cat_names = cat_names[:int(0.8 * cat_size)]
            elif part == 'test':
                cat_names = cat_names[int(0.8 * cat_size):]
            self.model_names += list(map(
                lambda n: '{}ShapeNetVox32/{}/{}'.format(path2data, cat, n),
                cat_names
            ))

    def __len__(self):
        return len(self.model_names)

    def __getitem__(self, i):
        with open('{}/model.binvox'.format(self.model_names[i]), 'rb') as model_file:
            line = model_file.readline().strip()
            if not line.startswith(b'#binvox'):
                raise IOError('Not a binvox file')
            line = model_file.readline()
            translate = np.array([float(j) for j in model_file.readline().strip().split(b' ')[1:]], dtype=np.float32)
            scale = np.array([float(j) for j in model_file.readline().strip().split(b' ')[1:]][0], dtype=np.float32)
            line = model_file.readline()
            raw_data = np.frombuffer(model_file.read(), dtype=np.uint8)
            values, counts = raw_data[::2], raw_data[1::2]
            grid = np.repeat(values, counts).astype(np.uint8)
            grid = grid.reshape(32, 32, 32)

        return grid
