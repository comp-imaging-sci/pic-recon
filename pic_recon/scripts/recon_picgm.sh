#!/bin/bash

cd ../src/

process=0
data_type=brain # faces or brain
mask_type=mask_rand_6x # gaussian_0.02, mask_rand_6x, mask_rand_8x
snr=20
step=0.5
niter=10000

p1=2
p2=4
lamda_w=3e-10
# Regularization parameter chart:
# faces 50 fold gaussian : p1=2; p2=3; lamda_w=1e-09
# brain  6 fold MRI      : p1=2; p2=4; lamda_w=3e-10
# brain 12 fold MRI      : p1=3; p2=4; lamda_w=1e-13
# Different (p1, p2) values may change results. Useful values of p1, p2 lie either in the range [2,5] or in the range [6,10] 

network_path=../../stylegan2/nets/stylegan2-CompMRIT1T2-config-f.pkl

gt_filename=../ground_truths/${data_type}/xgt_$process.npy
pi_filename=../prior_images/${data_type}/xpi_$process.npy
fileroot=${mask_type}_picgm_level${p1}-${p2}_lamw${lamda_w}_step${step}_${niter}_${snr}SNR_$process
savedir=../results/${data_type}/${process}/picgm_${data_type}_${mask_type}_SNR${snr}

mkdir -p $savedir

python -u recon_picgm_extended.py --network_path $network_path --data_type $data_type --mask_type $mask_type --snr $snr --savedir $savedir --mode simulation --niter $niter --logfile $savedir/log.log --lamda_w $lamda_w --step $step --gt_filename $gt_filename --pi_filename $pi_filename --cutoff_levels $p1 $p2 --snr $snr --fileroot $fileroot --extend --smart_init # --input_shape 1 128 128 3

# For face images, switch on "--input_shape 1 128 128 3"

