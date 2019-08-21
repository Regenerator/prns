import argparse
import numpy as np
import h5py as h5

from sys import stdout
from torch.utils.data import DataLoader

from lib.preprocessing.datasets import AllImagesDataset, AllGridDataset


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Data processor for ShapeNetAll dataset. '
        'All the images are accumulated in a single .h5 file. '
        '3D voxel occupancy grids are turned into bitpacks and are accumulated in another .h5 file. '
    )
    parser.add_argument('data', help='Path to dataset (should contain ./ShapeNetRendering and ./ShapeNetVox32)')
    parser.add_argument('save', help='Path to directory to save the file')
    parser.add_argument('images', type=np.int, nargs='?', default=1,
                        help='Signal to process/not process images')
    parser.add_argument('grids', type=np.int, nargs='?', default=1,
                        help='Signal to process/not process grids')
    parser.add_argument('bs', type=np.int, nargs='?', default=128,
                        help='Data loader batch size')
    parser.add_argument('nw', type=np.int, nargs='?', default=8,
                        help='Data loader number of workers')
    return parser


def process_images(path2data, path2save, batch_size=128, num_workers=8):
    train_images_dataset = AllImagesDataset(path2data, part='train')
    test_images_dataset = AllImagesDataset(path2data, part='test')
    train_images_iterator = DataLoader(train_images_dataset,
                                       batch_size=batch_size,
                                       num_workers=num_workers)
    test_images_iterator = DataLoader(test_images_dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers)

    fout = h5.File(path2save + '/ShapeNet_All_images.h5', 'w')
    train_images = fout.create_dataset('train_images',
                                       shape=(len(train_images_dataset), 4, 137, 137),
                                       dtype=np.uint8)
    test_images = fout.create_dataset('test_images',
                                      shape=(len(test_images_dataset), 4, 137, 137),
                                      dtype=np.uint8)

    print('Packing train images: ')
    for i, batch in enumerate(train_images_iterator):
        train_images[batch_size * i:batch_size * (i + 1)] = batch.numpy()
        stdout.write('{}/{}\n'.format(i + 1, len(train_images_iterator)))
        stdout.flush()
    print('Done!')

    print('Packing test images: ')
    for i, batch in enumerate(test_images_iterator):
        test_images[batch_size * i:batch_size * (i + 1)] = batch.numpy()
        stdout.write('{}/{}\n'.format(i + 1, len(test_images_iterator)))
        stdout.flush()
    print('Done!')

    fout.close()


def process_grids(iterator, grids, batch_size):
    print('Packing {} grids:'.format(iterator.dataset.part))
    for i, batch in enumerate(iterator):
        grids_batch = batch.numpy().astype(np.uint8)
        bitpacks_batch = np.packbits(grids_batch.reshape(grids_batch.shape[0], -1), axis=1)
        grids[batch_size * i:batch_size * (i + 1)] = bitpacks_batch
        stdout.write('{}/{}\n'.format(i + 1, len(iterator)))
        stdout.flush()
    print('Done!')


def process_data(path2data, path2save, grid_size=32, batch_size=128, num_workers=8):
    train_grids_dataset = AllGridDataset(path2data, part='train')
    train_grids_iterator = DataLoader(train_grids_dataset, batch_size=batch_size, num_workers=num_workers)

    test_grids_dataset = AllGridDataset(path2data, part='test')
    test_grids_iterator = DataLoader(test_grids_dataset, batch_size=batch_size, num_workers=num_workers)

    print('Packing full grids:')
    fout = h5.File(path2save + '/ShapeNet_All_grids_full.h5', 'w')

    train_grids = fout.create_dataset(
        'train_grids',
        shape=(len(train_grids_dataset), grid_size**3 // 8),
        dtype=np.uint8
    )
    process_grids(train_grids_iterator, train_grids, batch_size)

    test_grids = fout.create_dataset(
        'test_grids',
        shape=(len(test_grids_dataset), grid_size**3 // 8),
        dtype=np.uint8
    )
    process_grids(test_grids_iterator, test_grids, batch_size)

    fout.close()


parser = define_options_parser()
args = parser.parse_args()
if args.images == 1:
    process_images(args.data, args.save,
                   batch_size=args.bs, num_workers=args.nw)

if args.grids == 1:
    process_data(args.data, args.save, grid_size=32,
                 batch_size=args.bs, num_workers=args.nw)
