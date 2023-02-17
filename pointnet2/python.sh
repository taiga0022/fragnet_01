#!/bin/bash

## Move to current directory
#cd $PBS_O_WORKDIR

cd tsato/environmental1
source VENV_test1/bin/activate

cd Fragnet_newfps_yoshino
python3 pointnet2/train.py task=frag gpus=[0,1,2,3] batch_size=16