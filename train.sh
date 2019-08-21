#!/bin/bash -l

python train_CVAE.py ./configs/ShapeNet_All/cvae_latent.yaml cvae_latent False 8 0.000256 0.9 0.995 0.000008
python train_CVAE.py ./configs/ShapeNet_All/cvae_latent.yaml cvae_latent True 12 0.000064 0.9 0.99 0.000004
python train_CVAE.py ./configs/ShapeNet_All/cvae_latent.yaml cvae_latent True 16 0.000016 0.9 0.95 0.000002
python train_CVAE.py ./configs/ShapeNet_All/cvae_latent.yaml cvae_latent True 17 0.000004 0.9 0.9 0.000001
