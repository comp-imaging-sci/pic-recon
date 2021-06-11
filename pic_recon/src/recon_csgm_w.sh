#!/bin/bash

process=0
data_type=brain # faces or brain
mask_type=mask_rand_6x # gaussian_0.02, mask_rand_6x, mask_rand_8x
snr=20
niter=10000
step=0.5
lamda_w=0

network_path=../../stylegan2/nets/stylegan2-FastMRIT1T2-config-h.pkl

gt_filename=../ground_truths/${data_type}/xgt_$process.npy
savedir=../results/${data_type}/${process}/csgm_${data_type}_${mask_type}_SNR${snr}
fileroot=${mask_type}_csgm_step${step}_lamw${lamda_w}_${niter}_${snr}SNR_$process

mkdir -p $savedir

python -u recon_csgm_w.py --network_path $network_path --data_type $data_type --mask_type $mask_type --snr $snr --savedir $savedir --mode simulation --niter $niter --logfile $savedir/log.log --gt_filename $gt_filename --step $step --lamda_w $lamda_w --fileroot $fileroot --extend --smart_init # --image_shape 128 128 3 

# for faces switch on "--image_shape 128 128 3 "
