#!/bin/bash

# Activate Python venv
source /opt/aws_neuron_venv_pytorch/bin/activate

# Install Jupyter notebook kernel
# pip install ipykernel
# python3.8 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
# pip install jupyter notebook
# pip install environment_kernels

# Set pip repository pointing to the Neuron repository
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# The dependency path does not have the permissions to install with the default user.
# This is a workaround to install the dependencies
sudo chmod -R 777 /opt/aws_neuron_venv_pytorch/*
# Install transformers
python -m pip install transformers-neuronx sentencepiece
# Update Neuron Compiler and Framework
# python -m pip install --upgrade neuronx-cc==2.* torch-neuronx torchvision
