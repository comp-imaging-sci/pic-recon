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
parser.add_argument("--logfile", type=str, default='', help="Path to logfile")
parser.add_argument("--step", type=float, default=1.e-3, help="Step size")
parser.add_argument("--lamda", type=float, default=0., help="Lamda value")
parser.add_argument("--lamda_w", type=float, default=0., help="Regularization strength for w")
parser.add_argument("--tv", type=float, default=0, help="TV regularization parameter")
parser.add_argument("--optim_varnames", type=str, nargs='+', default=['w'], help="Variables over which to optimize (can be w or z and/or zn)")
parser.add_argument("--image_shape", nargs='+', type=int, default=[256,256,1], help="Image shape")
parser.add_argument("--niter", type=int, default=20000, help="niter")
parser.add_argument("--sampling_seed", type=int, default=0, help="Seed for sampling the ground truth for the inverse crime case")
parser.add_argument("--mode", type=str, default='inverse_crime', help="Whether or not to do an inverse crime study.")
parser.add_argument("--gt_filename", type=str, default='', help="path to the ground truth x")
parser.add_argument("--snr", type=float, default=20., help="SNR")
parser.add_argument("--extend", action='store_true', help="Extend w space")
parser.add_argument("--fileroot", type=str, default='', help="Fileroot to use")
parser.add_argument("--smart_init", action='store_true', help="Smart initialization based on the value of the loss function")
parser.add_argument("--network_path", type=str, default='')
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

if args.data_type=='brain' or args.data_type=='faces' or args.data_type=='brats':
    logdir = args.network_path
print(logdir)

# ---

sys.path.append('../../')
sys.path.append('../../stylegan2')
import importlib
import tensorflow as tf
import numpy as np
import pickle
import stylegan2.model_oop as sgan
import forward_models;from forward_models import *
import inverse_problems;from inverse_problems import *
importlib.reload(forward_models);from forward_models import *
importlib.reload(inverse_problems);from inverse_problems import *
import utils;from utils import *
importlib.reload(utils);from utils import *
import scipy.linalg as la
import imageio as io

# load the generative model
sess = sess = tf.Session().__enter__()
gen = sgan.StyleGAN(logdir, sess=sess)

DIM = args.image_shape[1]

# ground truth
if args.mode == 'inverse_crime' or args.mode == 'simulation':
    gt_basename = os.path.splitext(os.path.basename(args.gt_filename))[0][1:]
    xnp = np.load(args.gt_filename)
    xgt_path = os.path.join(args.savedir, f"x_gt_{args.data_type}")
    xgt_path = os.path.join(args.savedir, 'x'+gt_basename)
    np.save(xgt_path+'.npy', xnp); io.imsave(xgt_path+'.png', np.squeeze(xnp))
else:
    raise NotImplementedError()

rnd = np.random.RandomState(seed=1234)
# z_init = np.load('zinit_csgm.npy')

# Forward model
# Gaussian
if 'gaussian' in args.mask_type:
    samp = float(args.mask_type.split('_')[1])
    shapex = [1,*args.image_shape]
    shapey = [1, int(np.prod(shapex)*samp)]
    fwd_op = GaussianSensor(shapey, shapex)
    data_dtype = tf.float32

elif 'mask' in args.mask_type:
    # Fourier undersampling
    shapex = [1, *args.image_shape]
    assert shapex[-1] == 1, "Fourier undersampling only works for single channel images."
    shapey = shapex[:-1]
    center_fractions = [0.1]
    accelerations = [8] # center_fractions and accelerations dont matter for poisson disc sampling
    fwd_op = MRISubsampler(shapey, shapex, center_fractions, accelerations, loadfile=f'../masks_mri/{args.mask_type}.npy', is_complex=(args.data_type=='experimental'))
    data_dtype = tf.complex64

# solver
solver = RegularizedSolverBase(gen, fwd_op, 
    optim_varnames          = args.optim_varnames, 
    regularization_types    = {'w':'l2w'},
    dim_out                 = DIM,
    data_dtype              = data_dtype)

# Get measurement
y_meas = fwd_op._np(xnp)
# NOTE: CAUTION: I AM CURRENTLY USING THE ORTHO MODE IN FFT HENCE CAN GET AWAY WITH CALCULATING POWER OF xnp 
power_adjustment = 10.*np.log10(la.norm(xnp)**2/la.norm(y_meas)**2)
SNR = args.snr - power_adjustment*('mask' in args.mask_type)
noise_mode = 'complex' if 'mask' in args.mask_type else 'gaussian'
y_meas = utils.add_noise(y_meas, SNR, mode=noise_mode, seed=42)

# Initialization
_,_,w_init,zn_init = gen.sample(temp=0)
w_init = np.stack([gen.wavg]*w_init.shape[1], axis=0).reshape(1,w_init.shape[1],512)

if args.smart_init:
    # if os.path.exists(os.path.join(args.savedir, 'w_best_init.npy')): pass # w_init = np.load(os.path.join(args.savedir, 'w_best_init.npy'))
    # else:
    _,wrand = gen.sample_w(batch_size=100, temp=1, seed=789)
    losses = []
    for i,wr in enumerate(wrand):
        ww = wrand[i:i+1]
        x_init = gen(ww, zn_init, use_latent='w', Numpy=True, dim_out=DIM)
        losses.append( la.norm(y_meas - fwd_op._np(x_init)) )
        print(i, losses[-1])
    losses = np.array(losses)
    best_init_idx = np.where(losses==losses.min())[0]
    print(f"Picking {best_init_idx}")
    w_init[0] = wrand[best_init_idx]
    np.save(os.path.join(args.savedir, 'w_best_init.npy'), w_init)    

zn_init = gen.latent_coeffs_to_array(zn_init)*0



basename = args.fileroot
xestname = os.path.join(args.savedir, 'xest_'+basename+'.npy')
lestname = os.path.join(args.savedir, 'lest_'+basename+'.npy')
imgname  = os.path.join(args.savedir, 'xest_'+basename+'.png')
print(xestname)

print(args.lamda_w, args.lamda, args.niter, args.step)
# xest,west,znst = solver.fit(y_meas,
#     step                    = args.step,
#     lamdas                  = {'w':args.lamda_w, 'zn': args.lamda},
#     inits                   = {'w': w_init, 'zn': zn_init},
#     n_iter                  = args.niter,
#     scheduling              = 'adam',
#     step_schedule_params    = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
#     extend                  = args.extend,
#     check_recon_error       = True,
#     ground_truth            = xnp)

xest,west,znst = solver.fit(y_meas,
    step                    = args.step,
    lamdas                  = {'w':1e-10, 'zn': args.lamda},
    inits                   = {'w': w_init, 'zn': zn_init},
    n_iter                  = args.niter,
    scheduling              = 'adam',
    step_schedule_params    = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
    extend                  = args.extend,
    check_recon_error       = True,
    ground_truth            = xnp)

np.save(xestname, xest)
np.save(lestname, {'w':west, 'zn':znst})
io.imsave(imgname, xest[0])
