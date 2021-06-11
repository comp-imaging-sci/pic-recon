""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str, default='brain', 
                        help="Data type on which the network is trained: knee/knee-ood/brain/brain-ood/cpx/cpx-ood")
parser.add_argument("--mask_type", type=str, default='cartesian_4x', help="MRI subsampling rate")
parser.add_argument("--savedir", type=str, default='../results/', help="Folder to save the results")
parser.add_argument("--network_path", type=str, default='')
parser.add_argument("--logfile", type=str, default='', help="Path to logfile")
parser.add_argument("--step", type=float, default=1.e-3, help="Step size")
parser.add_argument("--lamda", type=float, default=0., help="Lamda value")
parser.add_argument("--lamda_w", type=float, default=0., help="Lamda value for w")
parser.add_argument("--alpha", type=float, default=0., help="Alpha value for PIC reg.")
parser.add_argument("--tv", type=float, default=0, help="TV regularization parameter")
parser.add_argument("--niter", type=int, default=20000, help="niter")
parser.add_argument("--sampling_seed", type=int, default=0, help="Seed for sampling the ground truth for the inverse crime case")
parser.add_argument("--mode", type=str, default='inverse_crime', help="Whether or not to do an inverse crime study.")
parser.add_argument("--ablation", action='store_true')
parser.add_argument("--input_shape", type=int, nargs='+', default=[1,256,256,1], help="Shape of x")
parser.add_argument("--num_points", default=8, type=int, help="Number of points during parameter sweep")
parser.add_argument("--snr", type=float, default=20., help="SNR")
parser.add_argument("--cutoff_levels", type=int, default=[0,14], nargs='+', help="Cutoff levels at which to do style mixing")
parser.add_argument("--optim_varnames", type=str, nargs='+', default=['w'], help="Variables over which to optimize (can be w or z and/or zn)")
# parser.add_argument("--pic_method", type=str, default='extended', help="Use PIC recon solver 1 or 2 or 3 or noinv")
parser.add_argument("--gt_filename", type=str, default='', help="path to the ground truth x")
parser.add_argument("--pi_filename", type=str, default='', help="path to the prior image x")
parser.add_argument("--extend", action='store_true', help="Extend w space")
parser.add_argument("--smart_init", action='store_true', help="Smart initialization based on the value of the loss function")
parser.add_argument("--fileroot", type=str, default='', help="file root")
args = parser.parse_args()

logfile = args.logfile
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()
print(args, flush=True)

# ---

logdir = args.network_path
print(logdir)

# ---

import importlib
sys.path.append('../../')
if 'stylegan2-ada' in args.network_path:
    sys.path.append('../../stylegan2-ada')
    import model_oop as sgan
else:
    sys.path.append('../../stylegan2')
    import stylegan2.model_oop as sgan
import importlib
import tensorflow as tf
import numpy as np
import pickle
import os
from forward_models import *
from inverse_problems import *
import utils
import scipy.linalg as la
import imageio as io

# load the generative model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = sess = tf.Session(config=config).__enter__()
gen = sgan.StyleGAN(logdir, sess=sess)

DIM = args.input_shape[1]

# ground truth
if args.mode == 'inverse_crime':
    gt_basename = os.path.splitext(os.path.basename(args.gt_filename))[0][1:]
    xnp = np.load(args.gt_filename)
    xgt_path = os.path.join(args.savedir, 'x'+gt_basename)
    np.save(xgt_path+'.npy', xnp); io.imsave(xgt_path+'.png', np.squeeze(xnp))

    latents_name = os.path.split(args.gt_filename)
    latents_name = os.path.join( latents_name[0], 'l' + latents_name[1][1:])
    latents = np.load(latents_name, allow_pickle=True).item()
    wnp = latents['w']; znnp = latents['zn']    
    latents_path = os.path.join(args.savedir, 'l'+gt_basename)
    np.save(latents_path+'.npy', latents)

elif args.mode == 'simulation':
    gt_basename = os.path.splitext(os.path.basename(args.gt_filename))[0][1:]
    xnp = np.load(args.gt_filename)
    xgt_path = os.path.join(args.savedir, 'x'+gt_basename)
    np.save(xgt_path+'.npy', xnp); io.imsave(xgt_path+'.png', np.squeeze(xnp))

pi_basename = os.path.splitext(os.path.basename(args.pi_filename))[0][1:]
x_prior = np.load(args.pi_filename)
latents_name = os.path.split(args.pi_filename)
latents_name = os.path.join( latents_name[0], 'l' + latents_name[1][1:])
latents = np.load(latents_name, allow_pickle=True).item()
w_prior = latents['w']; zn_prior = latents['zn']
x_prior_path = os.path.join(args.savedir, 'x'+pi_basename)
x_prior_g = gen(w_prior, zn_prior, Numpy=True, use_latent='w', dim_out=DIM)
np.save(x_prior_path+'.npy', x_prior_g); io.imsave(x_prior_path+'.png', np.squeeze(x_prior_g))
latents_path = os.path.join(args.savedir, 'l'+pi_basename)
np.save(latents_path+'.npy', latents)

rnd = np.random.RandomState(seed=1234)
# z_init = np.load('zinit_csgm.npy')

# Forward model
# Gaussian
if 'gaussian' in args.mask_type:
    samp = float(args.mask_type.split('_')[1])
    shapex = args.input_shape
    shapey = [1, int(np.prod(shapex)*samp)]
    fwd_op = GaussianSensor(shapey, shapex, assign_on_the_fly=(samp>=0.2))
    data_dtype = tf.float32
elif 'mask' in args.mask_type:
    # # Variable disc Poisson sampler
    shapey = [1,256,256]
    center_fractions = [0.1]
    accelerations = [8] # center_fractions and accelerations dont matter for poisson disc sampling
    fwd_op = MRISubsampler(shapey, args.input_shape, center_fractions, accelerations, loadfile=f'../masks_mri/{args.mask_type}.npy', is_complex=(args.data_type=='experimental'))
    data_dtype = tf.complex64

# solver
solver = PICGMSolver(gen, fwd_op,
    optim_varnames       = args.optim_varnames,
    regularization_types = {'w': 'l2w'},
    dim_out              = DIM,
    data_dtype           = data_dtype)

# Get measurement
y_meas = fwd_op._np(xnp)
# NOTE: CAUTION: I AM CURRENTLY USING THE ORTHO MODE IN FFT HENCE CAN GET AWAY WITH CALCULATING POWER OF xnp 
power_adjustment = 10.*np.log10(la.norm(xnp)**2/la.norm(y_meas)**2)
SNR = args.snr - power_adjustment * ('mask' in args.mask_type)
noise_mode = 'complex' if 'mask' in args.mask_type else 'gaussian'
y_meas = utils.add_noise(y_meas, SNR, mode=noise_mode, seed=42)
np.save(os.path.join(args.savedir, f'y_meas_{SNR}SNR'+ gt_basename[2:] +'.npy'), y_meas)

# Initialization
if args.mode == 'inverse_crime':
    zn_init = gen.latent_coeffs_to_array(zn_prior)
elif args.mode == 'simulation':
    _,_,_,zn_init = gen.sample(temp=0)
w_init = np.zeros(gen.shapew); # w_init[:,:] = gen.wavg
w_init = w_prior.copy()
# w_init[:,7:9] = np.load('brats_t2_latent.npy', allow_pickle=True).item()['w'][:,7:9]
# w_init[:,3:5] = gen.wavg

if args.smart_init:
    # if os.path.exists(os.path.join(args.savedir, 'w_best_init.npy')): pass # w_init = np.load(os.path.join(args.savedir, 'w_best_init.npy'))
    # else:
    _,wrand = gen.sample_w(batch_size=100, temp=1, seed=789)
    wrand = np.concatenate([wrand, w_prior], axis=0)
    ww = w_init.copy()
    losses = []
    for i,wr in enumerate(wrand):
        ww[:, args.cutoff_levels[0] : args.cutoff_levels[1] ] = wr[ args.cutoff_levels[0] : args.cutoff_levels[1] ]
        x_init = gen(ww, zn_init, use_latent='w', Numpy=True, dim_out=DIM)
        losses.append( la.norm(y_meas - fwd_op._np(x_init)) )
        print(i, losses[-1])
    losses = np.array(losses)
    best_init_idx = np.where(losses==losses.min())[0]
    print(f"Picking {best_init_idx}")
    w_init[:, args.cutoff_levels[0] : args.cutoff_levels[1] ] = wrand[best_init_idx, args.cutoff_levels[0] : args.cutoff_levels[1] ]
    np.save(os.path.join(args.savedir, 'w_best_init.npy'), w_init)    


zn_init = gen.latent_coeffs_to_array(zn_init)*0

if args.ablation:
    lamda_ws = list(10**(4.*np.linspace(0.,1.,args.num_points) + np.log10(args.lamda_w)))    
    basename = f'{args.mask_type}_picgm_level{args.cutoff_levels[0]}-{args.cutoff_levels[1]}_lam{args.lamda}_lamw{{}}_tv{args.tv}_step{args.step}_{args.niter}_{args.snr}SNR'
    basenames = [basename.format(l) for l in lamda_ws]
else:
    lamda_ws = [args.lamda_w]
    basenames = [args.fileroot]

for basename, lamda_w in zip(basenames, lamda_ws):
    xestname = os.path.join(args.savedir, 'xest_'+basename+'.npy')
    lestname = os.path.join(args.savedir, 'lest_'+basename+'.npy')
    imgname  = os.path.join(args.savedir, 'xest_'+basename+'.png')
    print(xestname)
    xest,west,znst = solver.fit(y_meas,
        step                    = args.step,
        lamdas                  = {'w':lamda_w, 'zn': args.lamda},
        inits                   = {'w': w_init, 'zn': zn_init},
        n_iter                  = args.niter,
        scheduling              = 'adam',
        w_prior_image           = w_prior,
        cutoff_levels           = args.cutoff_levels,
        step_schedule_params    = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        extend                  = args.extend,
        check_recon_error       = True,
        ground_truth            = xnp)

    np.save(xestname, np.clip(xest, -1,1))
    np.save(lestname, {'w': west, 'zn': znst})
    io.imsave(imgname, np.squeeze(xest))