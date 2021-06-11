#!/bin/bash

cd ../src/

process=0
data_type=brain # faces or brain
mask_type=mask_rand_6x # gaussian_0.02, mask_rand_6x, mask_rand_8x
snr=20
niter=2000
step=1.0

lamda=0.052
# Regularization parameter chart:
# faces 50 fold gaussian : 0.02
# brain  6 fold MRI      : 0.052
# brain 12 fold MRI      : 0.07

savedir=../results/${data_type}/${process}/fista_${data_type}_${mask_type}_SNR${snr}
gt_filename=../ground_truths/${data_type}/xgt_$process.npy

fileroot=xest_fista_${mask_type}_step${step}_tv${lamda}_${niter}_${snr}SNR_${process}

mkdir -p $savedir

python -u recon_fista.py --process $process --data_type $data_type --mask_type $mask_type --lamda $lamda --savedir $savedir --logfile $savedir/log.log --niter $niter --snr $snr --gt_filename $gt_filename --step $step --fileroot $fileroot # --image_shape 128 128 3

# For faces, switch on "--image_shape 128 128 3"
