#!/bin/bash

module load cuda-toolkit/10
module load gcc/7.2.0

network=nets/stylegan2-FastMRIT1T2-config-h.pkl
data_dir=../pic_recon/prior_images/
result_dir=projected_images/FastMRIT1T2-config-h

python -u run_projector.py project-real-images-from-npy --network=$network --data-dir $data_dir --num-images 2 --mode normal --result-dir $result_dir
