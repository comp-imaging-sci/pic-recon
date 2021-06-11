#!/bin/bash

process=0
data_type=brain # faces or brain
mask_type=mask_rand_6x # gaussian_0.02, mask_rand_6x, mask_rand_8x
snr=20
niter=2000
step=1.0

lamda=0.0072
alpha=0.5
# Regularization parameter chart:
# faces 50 fold gaussian : lamda=0.0517; alpha=0.5 (alpha varies a lot based on the image, try other values too)
# brain  6 fold MRI      : lamda=0.0072; alpha=0.5
# brain 12 fold MRI      : lamda=0.0072; alpha=0.5

savedir=../results/${data_type}/${process}/piccs_${data_type}_${mask_type}_SNR${snr}
gt_filename=../ground_truths/${data_type}/xgt_$process.npy
pi_filename=../prior_images/${data_type}/xpi_$process.npy

fileroot=xest_piccs_${mask_type}_lr${step}_tv${lamda}_pi${alpha}_${niter}_${snr}SNR_${process}

mkdir -p $savedir

python -u recon_piccs.py --process $process --data_type $data_type --mask_type $mask_type --lamda $lamda --alpha $alpha --savedir $savedir --logfile $savedir/log.log --niter $niter --snr $snr --gt_filename $gt_filename --pi_filename $pi_filename --fileroot $fileroot --step $step # --input_shape 1 128 128 3

# For face images, switch on "--input_shape 1 128 128 3"