import json
try:
    import tensorflow as tf
    # import horovod.tensorflow as hvd
except:
    print("Not importing tensorflow. This may lead to errors while using functions:")
    print("tensorflow_session(), get_its()")
import numpy as np
import os
import argparse
from PIL import Image
import scipy.linalg as la
import scipy.sparse.linalg as sla
try:
    import prox_tv as tv
except ModuleNotFoundError:
    print("ProxTV not available. CAUTION")

# for wavelet transforms
try:
    import pywt
    WAVELET = pywt.Wavelet('coif3')
    LEVEL = 3
except:
    print("Warning: pywt not imported. Wavelet transforms will not work if used.")


class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        self.f_log = open(path, 'w')
        self.f_log.write(json.dumps(kwargs) + '\n')

    def log(self, **kwargs):
        self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()


def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess


def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


def coalesce(xs):
    return np.concatenate(xs, axis=1)


def imsave(filename, img):
    im = Image.fromarray(img)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(filename)
    print("Saved to "+filename)


def get_hps(logdir):
    filename = os.path.join(logdir, 'test.txt')
    with open(filename, 'r') as file:

        hps_dict = json.loads(file.readline())
        return argparse.Namespace(**hps_dict)


def add_noise(x, SNR, mode='gaussian', seed=None):
    """ Adds gaussian noise of a given SNR to a signal
    """

    rnd = np.random if (seed == None) else np.random.RandomState(seed)

    p_signal = la.norm(x)**2
    snr_inv = 10**(-0.1*SNR)

    p_noise = p_signal * snr_inv
    sigma = np.sqrt( p_noise/np.prod(x.shape) )

    if mode=='gaussian':
        x_noisy = x + sigma * rnd.randn(*(x.shape))
    elif mode=='salt_pepper':
        x_noisy = x + sigma * abs(rnd.randn(*(x.shape)))
    elif mode=='complex':
        x_noisy = x + sigma/np.sqrt(2) * (rnd.randn(*(x.shape)) + 1.j*rnd.randn(*(x.shape)))
    else:
        raise ValueError("Enter a suitable mode")

    return x_noisy.astype(x.dtype)


# Taken from https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    import warnings
    import functools
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def psnr(gt, recon, max_val=255):
    mse = la.norm(gt-recon)**2/np.prod(gt.shape)
    return 10 * np.log10( max_val**2 / mse)


def soft(x, lam, weights=None):
    """ Simple pointwise soft-thresholding.
    Inputs:
    `x`         : Numpy array to be soft thresholded.
    `lam`       : Soft threshold
    `weights`   : Weighting on soft threshold per coefficient for weighted L1 norm.
    """
    if weights==None:
        weights = np.ones_like(x)
    lamda = lam / weights
    z = (x - lamda)*(x > lamda) + (x + lamda)*(x < -lamda)
    return z.astype(x.dtype)

def wavesoft(x,lamda):
    """ Wavelet transform soft thresholding Asumes a 2d numpy array as input.
    """
    coeffs = pywt.wavedec2(x, wavelet=WAVELET, level=LEVEL)
    c, slices = pywt.coeffs_to_array(coeffs)
    c = soft(c, lamda)
    coeffs = pywt.array_to_coeffs(c, slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, WAVELET)

def tv1_2d_cpx(x, lam):
    if x.dtype in [complex, np.complex64, np.complex128]:
        z = tv.tv1_2d(x.real, lam) + 1.j*tv.tv1_2d(x.imag, lam)
    else:
        z = tv.tv1_2d(x, lam)
    return z.astype(x.dtype)


def tf_inprod(x1, x2):
    """ Takes inner product of two real tensorflow tensors
    """
    return tf.reduce_sum( x1 * x2 )

def lipschitz(fwd):
    # create an abstract operator equivalent to A^T A (normal operator), where A = forward operator. 
    # Then calculate its highest eigenvalue to compute the lipschitz constant of A.
    mv = lambda ip: fwd.adj_np( fwd._np(ip.reshape(fwd.shapeOI[1])) ).reshape(fwd.shape[1])
    normal = sla.LinearOperator(
        (fwd.shape[1], fwd.shape[1]),
        matvec=mv,
    )
    return 2*sla.eigs(normal, k=1)[0].real


def latent_coeffs_to_array(zn):

    z = [ zz.reshape(zz.shape[0], -1) for zz in zn ]
    z = np.concatenate(z, axis=-1)
    return z

def latent_array_to_coeffs(z, max_dim):

    sections = [0]
    for i in range(2, int(np.log2(max_dim))+1):
        sections.append(
            sections[-1] + (2**i) **2
        )
        sections.append(
            sections[-1] + (2**i) **2
        )
    sections = sections[1:-1]
    zz = np.split(z, sections, axis=1)
    zz = iter(zz)
    zn = []

    for i in range(2, int(np.log2(max_dim))+1):
        zsingle = next(zz)
        zn.append(
            zsingle.reshape( zsingle.shape[0], 1, 2**i, 2**i )
        )
        zsingle = next(zz)
        zn.append(
            zsingle.reshape( zsingle.shape[0], 1, 2**i, 2**i )
        )

    return zn


