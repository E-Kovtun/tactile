#!/bin/bash -e
shopt -s expand_aliases
# if type -P micromamba; then
# 	echo "micromamba detect, using micromamba inplace of mamba"
# 	alias conda=micromamba
# 	eval "$(micromamba shell hook --shell bash)"
# elif type -P mamba || type -P conda; then
# 	eval "$(conda shell.bash hook)"
# else
# 	echo please install mamba or micromamba
# 	exit
# fi


# mamba create -y --name tactile_ssl python=3.10

# conda activate tactile_ssl

# Pytorch no longer provides conda packages
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# install xformers (make sure that the cuda versions are compatible)
python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu126

python -m pip install hydra-core hydra-colorlog wandb matplotlib einops tqdm scipy scikit-learn h5py rich seaborn scikit-learn moviepy
python -m pip install lightning

# install huggingface datasets
python -m pip install datasets safetensors fsspec requests pyyaml

python -m pip install sympy

python -m pip install rootutils opencv-python pytorch-kinematics gdown pre-commit
python -m pip install mcap-ros2-support mcap joblib
python -m pip install pandas tqdm