""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str, default='brain', 
                        help="Data type on which the network is trained: knee/knee-ood/brain/brain-ood/cpx/cpx-ood")
parser.add_argument("--mask_type", type=str, default='cartesian_4fold', help="MRI mask type")
parser.add_argument("--gt_filename", type=str, default="xgt_s.npy", help="Path to the ground truth")
parser.add_argument("--pi_filename", type=str, default="", help="Path to the prior image")
parser.add_argument("--meas_filename", type=str, default='', help="Path to measurements")
parser.add_argument("--mode", type=str, default='simulation')
parser.add_argument("--savedir", type=str, default='', help="Folder to save the results")
parser.add_argument("--logfile", type=str, default='', help="Path to logfile")
parser.add_argument("--ablation", action="store_true", help="Whether or not to ablate")
parser.add_argument("--lamda", type=float, default=1.e-3, help="Lamda value")
parser.add_argument("--alpha", type=float, default=0.5, help="Regularization due to the prior image.")
parser.add_argument("--step", type=float, default=0.5, help="Step size")
parser.add_argument("--niter", type=int, default=1000, help="niter")
parser.add_argument("--input_shape", type=int, nargs='+', default=[1,256,256,1], help="Shape of x")
parser.add_argument("--snr", type=float, default=20., help="Measurement SNR")
parser.add_argument("--numrand", type=int, default=12, help="Number of sweep points")
parser.add_argument("--fileroot", type=str, default="File root", help="Root of path to recon file")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--process", type=int, default=0)
args = parser.parse_args()

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(args.logfile, "a")

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

sys.path.append('../../')
sys.path.append('../../stylegan2')
from forward_models import *
from inverse_problems import *
import matplotlib.pyplot as plt
from utils import *
import scipy.linalg as la
import imageio as io


# Forward model
# Gaussian
if 'gaussian' in args.mask_type:
    samp = float(args.mask_type.split('_')[1])
    shapex = args.input_shape
    shapey = [1, int(np.prod(shapex)*samp)]
    fwd_op = GaussianSensor(shapey, shapex)

# MRI
elif 'mask' in args.mask_type:
    # # Variable disc Poisson sampler
    shapey = [1,256,256]
    center_fractions = [0.1]
    accelerations = [8] # center_fractions and accelerations dont matter for poisson disc sampling
    fwd_op = MRISubsampler(shapey, args.input_shape, center_fractions, accelerations, loadfile=f'../masks_mri/{args.mask_type}.npy', is_complex=(args.data_type=='experimental'))

# ground truth and measurements
if args.mode == 'simulation':
    xnp = np.load(args.gt_filename)
    xpi = np.load(args.pi_filename)

    # Get measurement
    y_meas = fwd_op._np(xnp)
    # NOTE: CAUTION: I AM CURRENTLY USING THE ORTHO MODE IN FFT HENCE CAN GET AWAY WITH CALCULATING POWER OF xnp 
    power_adjustment = 10.*np.log10(la.norm(xnp)**2/la.norm(y_meas)**2)
    SNR = args.snr - power_adjustment * ('mask' in args.mask_type)
    noise_mode = 'complex' if 'mask' in args.mask_type else 'gaussian'
    y_meas = utils.add_noise(y_meas, SNR, mode=noise_mode, seed=42)

# create hparams for ablation
numrand = args.numrand
rnd = np.random.RandomState(seed=1234)
lams = 10**(3.*np.linspace(0.,1.,numrand) + np.log10(args.lamda))

if args.ablation:
    hparams = lams
else:
    hparams = [args.lamda]
print(hparams)
[print(h) for h in hparams]
sys.stdout.flush()

# solver
solver = PICCSSolver(tf.compat.v1.Session(), fwd_op,
    mode                    = 'subgrad',
    regularization_types    = {'x':'tv', 'pi_err':'l1wt-haar-6'},
    data_dtype              = tf.complex64)

for lamda in hparams:

    step = args.step; n_iter = args.niter
    if not args.ablation:
        fileroot = args.fileroot        
    else:
        fileroot = f"xest_piccs_{args.mask_type}_lr{args.step}_tv{lamda}_pi{args.alpha}_{args.niter}_{args.snr}SNR_{args.process}"
    filename =  os.path.join(args.savedir, fileroot + '.npy')    
    imagename = os.path.join(args.savedir, fileroot + '.png')
    print(filename, flush=True)
    xest = solver.fit(y_meas, step=step, lamda=lamda, 
                alpha               = args.alpha,
                x_prior_image       = xpi,
                n_iter              = n_iter,
                scheduling          = 'adam',
                reg_scheduling      = True,
                check_recon_error   = True, 
                reg_schedule_param  = 1.,
                ground_truth        = xnp)
    np.save(filename, xest)
    io.imsave(imagename, np.squeeze(xest))
    print("Saved to ", imagename)
    sys.stdout.flush()
