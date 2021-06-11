#!/bin/bash

module load cuda-toolkit/10
module load gcc/7.2.0

network=nets/stylegan2-FastMRIT1T2-config-h.pkl

data_dir=jacobians_expr

result_dir=worst_linapprox_images/FastMRIT1T2

python -u run_lin_limit.py optimize-linlimit --network=$network --data-dir $data_dir --num-images 2 --mode normal --epsilon $1 --result-dir $result_dir
