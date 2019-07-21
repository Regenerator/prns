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
from lib.networks.utils import evaluate


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model evaluating script. Provide a suitable config.')
    parser.add_argument('config', help='Path to config file in YAML format.')
    parser.add_argument('modelname', help='Postfix to model name.')
    parser.add_argument('part', help='Part of dataset (train / val / test).')
    parser.add_argument('predict', help='Prediction mode flag.')
    parser.add_argument('save', help='Saving flag.')
    return parser


parser = define_options_parser()
args = parser.parse_args()
with io.open(args.config, 'r') as stream:
    config = yaml.load(stream)
config['model_name'] = '{0}.pkl'.format(args.modelname)
config['resume'] = True
config['batch_size'] = 24 // config['img_n_views']
if args.predict == 'True':
    config['prediction_mode'] = True
else:
    config['prediction_mode'] = False
if args.save == 'True':
    config['saving_mode'] = True
else:
    config['saving_mode'] = False
print('Configurations loaded.')

image_transform = ComposeImageTransformation(**config)
grid_transform = ComposeGridTransformation(**config)
eval_dataset = ShapeNetMultiviewDataset(config['path2data'], part=args.part,
                                        n_views_used=config['img_n_views'], views_shuffle=False,
                                        grid_size=config['grid_size'], sample_grids=True,
                                        image_transform=image_transform, grid_transform=grid_transform)
print('Dataset init: done.')

eval_iterator = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False,
                           num_workers=config['num_workers'], pin_memory=True, drop_last=False)
print('Iterator init: done.')

model = GNet(**config).cuda()
print('Model init: done.')

criterion = GNetLoss(**config).cuda()
print('Losses initialization: done.')

checkpoint = torch.load(config['path2data'] + 'models/' + config['model_name'])
model.load_state_dict(checkpoint['model_state'])
del(checkpoint)
print('Model {} loaded.'.format(config['path2data'] + 'models/' + config['model_name']))

evaluate(eval_iterator, model, criterion, **config)
