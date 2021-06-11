import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str, default='knee', 
                        help="Data type on which the network is trained: knee/knee-ood/brain/brain-ood/cpx/cpx-ood")
parser.add_argument("--mask_type", type=str, default='cartesian_4fold', help="MRI mask type")
parser.add_argument("--gt_filename", type=str, default="xgt_s.npy", help="Path to the ground truth")
parser.add_argument("--meas_filename", type=str, default='', help="Path to measurements")
parser.add_argument("--savedir", type=str, default='', help="Folder to save the results")
parser.add_argument("--logfile", type=str, default='', help="Path to logfile")
parser.add_argument("--ablation", action="store_true", help="Whether or not to ablate")
parser.add_argument("--lamda", type=float, default=1.e-3, help="Lamda value")
parser.add_argument("--step", type=float, default=0.5, help="Step size")
parser.add_argument("--niter", type=int, default=1000, help="niter")
parser.add_argument("--snr", type=float, default=20., help="Measurement SNR")
parser.add_argument("--numrand", type=int, default=12, help="Number of sweep points")
parser.add_argument("--fileroot", type=str, default="File root", help="Root of path to recon file")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--process", type=int, default=0)
parser.add_argument("--image_shape", type=int, nargs='+', default=[256,256,1], help="Shape of x")
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
    shapex = [1,*args.image_shape]
    shapey = [1, int(np.prod(shapex)*samp)]
    fwd_op = GaussianSensor(shapey, shapex, assign_on_the_fly=(samp>=0.2))
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

# ground truth and measurements
if args.data_type != 'experimental':
    xnp = np.load(args.gt_filename)

    # Get measurement
    y_meas = fwd_op._np(xnp)
    # NOTE: CAUTION: I AM CURRENTLY USING THE ORTHO MODE IN FFT HENCE CAN GET AWAY WITH CALCULATING POWER OF xnp 
    power_adjustment = 10.*np.log10(la.norm(xnp)**2/la.norm(y_meas)**2)
    SNR = args.snr
    if 'gaussian' in args.mask_type:
        y_meas = utils.add_noise(y_meas, SNR, seed=args.seed)
    else:
        y_meas = utils.add_noise(y_meas, SNR-power_adjustment, mode='complex', seed=args.seed)
        mask = np.load('../masks_mri/{}.npy'.format(args.mask_type))
        y_meas = np.fft.ifftshift(mask) * y_meas

else:
    # Load fully sampled measurements
    y_full = np.fromfile(args.meas_filename, np.complex64).reshape(256,256)
    maxval = np.load( os.path.join(os.path.dirname(args.meas_filename), 'vol_max.npy') )
    y_full = y_full / maxval
    print("Maxval = ", maxval)
    sgn = (-1)**np.arange(256)
    sgn = np.stack([sgn]*256, axis=0)
    sgn[1::2] = -sgn[1::2]
    y_full = np.fft.ifftshift(y_full) * sgn
    mask = np.load('../masks_mri/{}.npy'.format(args.mask_type))
    y_meas = np.fft.ifftshift(mask) * y_full
    y_meas = y_meas.reshape(1, *y_meas.shape)
    
    # get "ground truth" as the ifft of fully sampled measurements
    xnp = np.fft.ifft2(y_full, norm='ortho')
    xnp = xnp.reshape(1, *xnp.shape, 1)
    np.save(args.gt_filename, xnp)

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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
solver = ProximalSolver(tf.compat.v1.Session(config=config), fwd_op, mode='fista', data_dtype=tf.complex64)

for lamda in hparams:

    step = args.step; n_iter = args.niter
    if not args.ablation:
        fileroot = args.fileroot        
    else:
        fileroot = f"xest_fista_{args.mask_type}_lr{args.step}_tv{lamda}_{args.niter}_{args.snr}SNR_{args.process}"
    filename =  os.path.join(args.savedir, fileroot + '.npy')    
    imagename = os.path.join(args.savedir, fileroot + '.png')
    print(filename, flush=True)
    xest = solver.fit(y_meas, step=step, lamda=lamda,
                n_iter=n_iter,
                scheduling='fista',
                reg_scheme='tv',
                reg_scheduling=True,
                check_recon_error=True, 
    #             step_schedule_params={'beta1':0.999, 'beta2':1.e-4, 'epsil':1.e-6},
                reg_schedule_param=1.,
                ground_truth=xnp)
    np.save(filename, xest)
    io.imsave(imagename, np.squeeze(xest))
    print("Saved to ", imagename)
    sys.stdout.flush()
