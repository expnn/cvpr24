#!/usr/bin/env bash

conda create -n torch python=3.9 numpy jupyter scipy PyYAML astropy intervaltree
conda activate torch
conda install -y pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
