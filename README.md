# LRI
This repository contains the official implementation of LRI as described in the paper: [Interpretable Geometric Deep Learning via Learnable Randomness Injection]() by Siqi Miao, Yunan Luo, Mia Liu, and Pan Li.

## Introduction
In this work, we propose four scientific datasets with ground-truth interpretation labels to benchmark interpretable Geometric Deep Learning (GDL) methods, and we propose a general framework named Learnable Randomness Injection (LRI) to build inherently interpretable GDL models based on a broad range of GDL backbones.

## Installation
We have tested our code on `Python 3.9` with `PyTorch 1.12.1`, `PyG 2.0.4` and `CUDA 11.3`. Please follow the following steps to create a virtual environment and install the required packages.

Clone the repository:
```
git clone https://github.com/Graph-COM/LRI.git
cd LRI
```

Create a virtual environment:
```
conda create --name lri python=3.9 -y
conda activate lri
```

Install dependencies:
```
conda install -y pytorch==1.12.1 torchvision cudatoolkit=11.3 -c pytorch
pip install torch-scatter==2.0.9 torch-sparse==0.6.14 torch-cluster==1.6.0 torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install -r requirements.txt
```

## Datasets
All our datasets can be downloaded and processed automatically by running the scripts in `./src/datasets`. By default, the code will ask if the raw files and/or the processed files should be downloaded. For example, to download and process the `SynMol` dataset, simply run:
```
cd ./src/datasets
python synmol.py
```

All datasets are also available to manually download from Zenodo: https://doi.org/10.5281/zenodo.7265547.


## Reproduce Results
Use the following command to train a model:

```
cd ./src
python trainer.py --backbone [backbone_model] --dataset [dataset_name], --method [method_name]
```
`backbone_model` can be choosen from `dgcnn`, `pointtrans` and `egnn`.

`dataset_name` can be choosen from `actstract_2T`, `tau3mu`, `synmol` and `plbind`.

`method_name` can be choosen from `lri_bern`, `lri_gaussian`, `gradcam`, `gradgeo`, `bernmask`, `bernmask_p`, and `pointmask`.

By adding `--cuda [GPU_id]` to the command, the code will run on the specified GPU; adding `--seed [seed_number]` to the command, the code will run with the specified random seed.

The tuned hyperparameters for all backbone models and interpretation methods can be found in `./src/config`.
