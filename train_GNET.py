import argparse
import io
import yaml

import torch
from torch.utils.data import DataLoader

from lib.datasets import ShapeNetMultiviewDataset
from lib.transformations import ComposeImageTransformation
from lib.transformations import ComposeGridTransformation

from lib.networks.model import GNet
from lib.networks.losses import GNetLoss
from lib.networks.optimizers import Adam, LRUpdater
from lib.networks.utils import train


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('config', help='Path to config file in YAML format.')
    parser.add_argument('modelname', help='Postfix to model name.')
    parser.add_argument('resume', help='Control flag, showing if the training should continue (True or False).')
    parser.add_argument('n_epochs', help='Number of training epochs.')
    parser.add_argument('lr', help='Learining rate parameter value.')
    parser.add_argument('beta1', help='First moment accumulation parameter value.')
    parser.add_argument('beta2', help='Second moment accumulation parameter value.')
    parser.add_argument('cycle_length', help='Cycle length for cyclic learning.')
    parser.add_argument('wd', help='Weight decay parameter value.')
    return parser


parser = define_options_parser()
args = parser.parse_args()
with io.open(args.config, 'r') as stream:
    config = yaml.load(stream)
config['model_name'] = '{0}.pkl'.format(args.modelname)
config['resume'] = True if args.resume == 'True' else False
config['n_epochs'] = int(args.n_epochs)
config['min_lr'] = config['max_lr'] = float(args.lr)
config['beta1'] = float(args.beta1)
config['min_beta2'] = config['max_beta2'] = float(args.beta2)
config['cycle_length'] = int(args.cycle_length)
config['wd'] = float(args.wd)
print('Configurations loaded.')

image_transform = ComposeImageTransformation(**config)
grid_transform = ComposeGridTransformation(**config)
train_dataset = ShapeNetMultiviewDataset(config['path2data'], part='train',
                                         n_views_used=config['img_n_views'],
                                         views_shuffle=False if config['img_n_views'] == 1 else True,
                                         grid_size=config['grid_size'], sample_grids=True,
                                         image_transform=image_transform, grid_transform=grid_transform)
print('Dataset init: done.')

train_iterator = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
                            num_workers=config['num_workers'], pin_memory=True, drop_last=True)
print('Iterator init: done.')

model = GNet(**config).cuda()
print('Model init: done.')

criterion = GNetLoss(**config).cuda()
optimizer = Adam(model.parameters(), lr=config['max_lr'], weight_decay=config['wd'],
                 betas=(config['beta1'], config['max_beta2']), amsgrad=True)
scheduler = LRUpdater(len(train_iterator), **config)
print('Optimizer init: done.')

if not config['resume']:
    cur_epoch = 0
    cur_iter = 0
else:
    checkpoint = torch.load(config['path2data'] + 'models/' + config['model_name'])
    cur_epoch = checkpoint['epoch']
    cur_iter = checkpoint['iter']
    model.load_state_dict(checkpoint['model_state'])
    # optimizer.load_state_dict(checkpoint['optimizer_state'])
    del(checkpoint)
    print('Model {} loaded.'.format(config['path2data'] + 'models/' + config['model_name']))

for epoch in range(cur_epoch, config['n_epochs']):
    train(train_iterator, model, criterion, optimizer, scheduler, epoch, cur_iter, **config)
    cur_iter = 0
