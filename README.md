# Probabilistic Reconstruction Networks
This repository contains the code for the **"Probabilistic Reconstruction Networks for 3D Shape Inference from a Single Image"** [paper](https://arxiv.org/abs/1908.07475) accepted as Oral at British Machine Vision Conference 2019. It includes:
- a data processing script for the ShapeNet dataset (the data could be taken from [here](https://github.com/chrischoy/3D-R2N2));
- training and IoU evaluation scripts for all the variations of Probabilistic Reconstruction Networks.

## Environment
The code requires python-3.6 and these packages, (and was run using according versions):
- yaml-0.1.7
- numpy-1.14.2
- opencv3-3.1.0
- h5py-2.7.1
- pytorch-1.1.0
- torchvision-0.3.0

## Usage
### Data preparation
The data can be prepared for use with:
```
python process_ShapeNetAll.py /path_to_raw_data /path_for_processed_data
```
This script relies on `/path_to_raw_data` containing `ShapeNetRendering` and `ShapeNetVox32` directories and outputs two hdf5 files containing images and voxel grids to `/path_for_processed_data`. Additional options could be viewed with:
```
python process_ShapeNetAll.py --help
```

### Training
All the hyperparameters of training procedures are situated in separate .yaml files in `./configs/ShapeNet_All/`, feel free to explore it. You will need to change the `path2data` field in the config files and also have a `models` directory in your `path2data` in order to save the model checkpoints.

Here is a complete list of training scripts and config names used to produce models corresponding to the line numbers in Table 2 in the paper:
1. `train_GNET.py ./configs/ShapeNet_All/gnet_latent.yaml`
2. `train_GNET.py ./configs/ShapeNet_All/gnet_cond.yaml`
3. `train_GNET.py ./configs/ShapeNet_All/gnet_latent_cond.yaml`
4. `train_CVAE.py ./configs/ShapeNet_All/cvae_latent.yaml`
5. `train_CVAE.py ./configs/ShapeNet_All/cvae_latent_cond.yaml`
6. `train_CVAE.py ./configs/ShapeNet_All/cvae_latent_econd.yaml`
7. `train_TLN.py ./configs/ShapeNet_All/cvae_latent_det(TLN).yaml`
8. `train_DVAE.py ./configs/ShapeNet_All/dvae_latent.yaml`

Training procedure is equal for all the models and can be found in `train.sh`. This script is written for the best performing configuration of the PRNs (line 4), in order to use it with other configurations you should change the executable python file, use proper config file (for those use correspondances above or change the config files) and provide a unique model name.

Finally, to train the chosen model run:
```
./train.sh
```

### Evaluating
Class specific evaluation of the IoU metric for test part of the dataset could be done with:
```
./evaluate.sh
```
Evaluation is written for the best configuration of the model and could be changed, similarly to training script.

## Citation
```
@InProceedings{klokov19bmvc,
  author    = {R. Klokov and J. Verbeek and E. Boyer},
  title     = {Probabilistic Reconstruction Networks for 3D Shape Inference from a Single Image},
  booktitle = {BMVC},
  year      = {2019}
}
```
