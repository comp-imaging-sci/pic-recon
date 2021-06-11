""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

import sys
sys.path.append("../")
import numpy as np 
import tensorflow as tf
from fastMRI.common import subsample

class GaussianSensor(object):
    """ A random gaussian forward operator for simple comressed sensing experiments. This includes routines for both numpy calculations as well as lazy tensorflow/keras calculations.
    Attributes : 
    `value`       : Values of elements in the forward operator as a matrix.
    `shape`       : matrix shape of the linear operator
    `shapeOI`     : Output-Input shape of the linear operator
    """

    def __init__(self, shapey, shapex, assign_on_the_fly=False, is_complex=False, seed=1234):
        m = np.prod(shapey)
        n = np.prod(shapex)
        self.assign_on_the_fly = assign_on_the_fly

        if is_complex:  
            self.input_dtype = self.output_dtype = np.complex64
            rnd = np.random.RandomState(seed=seed)
            self.value = 1./np.sqrt(2) * (rnd.randn(m,n).astype(np.float32) + 1.j*rnd.randn(m,n).astype(np.float32))
        else:   
            self.input_dtype = self.output_dtype = np.float32
            self.value = np.random.RandomState(seed=seed).randn(m,n).astype(np.float32) / np.sqrt(m)

        self.shape = (m,n)
        self.shapeOI = (shapey, shapex)
        
        if assign_on_the_fly:
            self.value_pl = tf.placeholder(dtype=self.input_dtype, shape=self.shape)
            self.value_tf = tf.compat.v1.get_variable('value_tf', list(self.shape))
        else:
            self.value_tf = tf.compat.v1.Variable(initial_value=self.value, trainable=False)
            self.init = tf.variables_initializer([self.value_tf])
        # self.value_tf = tf.cast(self.value_tf, tf.float32)

    def _np(self, x):
        A = self.value

        y = A @ x.flatten()
        return y.reshape(self.shapeOI[0])

    def adj_np(self, y):
        A = self.value
        x = A.T @ y.flatten()
        return x.reshape(self.shapeOI[1])

    def __call__(self, x):
        x1 = tf.reshape(x, [-1,1])
        if self.assign_on_the_fly:
            y1 = tf.tensordot(self.value_tf.assign(self.value_pl), x1, axes=1)
        else:
            y1 = tf.tensordot(self.value_tf, x1, axes=1)
        return tf.reshape(y1, self.shapeOI[0])

    def adj(self, y):
        """ Depreciated. Not required. DO NOT USE!
        """
        # y1 = tf.reshape(y, [-1,1])
        # x1 = tf.tensordot( K.transpose(self.placeholder), y1, axes=1)
        # return tf.reshape(x1, self.shapeOI[1])
        pass


class MRISubsampler(object):
    """ Forward model corresponding to MRI subsampling.
    Attributes : 
    `value`       : Values of elements in the forward operator as a matrix.
    `shape`       : matrix shape of the linear operator
    `shapeOI`     : Output-Input shape of the linear operator
    """

    def __init__(self, shapey, shapex, center_fractions=[0.08], accelerations=[4], fftnorm='ortho', loadfile=None, assign_on_the_fly=False, is_complex=False):
        """ Args :
        `shapey` :
        `shapex` : 
        `center_fractions` : Fraction of columns to be fully sampled in the center
        `accelerations` : Speedup on the rest. Refer to https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
        `loadfile` : Load mask pattern from a file. If None, generate random lines mask
        """

        m = np.prod(shapey)
        n = np.prod(shapex)

        self.assign_on_the_fly = assign_on_the_fly
        self.shape = (m,n)
        self.shapeOI = (shapey, shapex)
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.fftnorm = fftnorm
        self.loadfile = loadfile
        if is_complex:
            self.input_dtype = np.complex64
        else:
            self.input_dtype = np.float32
        self.output_dtype = np.complex64

        if self.loadfile:
            self.value = np.load(self.loadfile).astype(np.complex64)
            self.value = np.fft.ifftshift(self.value)
        else:
            self.value = subsample.MaskFunc(center_fractions, accelerations)(shapex[:-1])
            self.value = np.fft.ifftshift(self.value).astype(np.complex64)
        
        if assign_on_the_fly:
            self.value_pl = tf.placeholder(dtype=tf.complex64, shape=self.value.shape)
            self.value_tf = tf.compat.v1.get_variable('value_tf', list(self.value.shape))
        else:
            self.value_tf = tf.compat.v1.Variable(initial_value=self.value, trainable=False)
            self.init = tf.compat.v1.variables_initializer([self.value_tf])
        # self.value_tf = tf.cast(self.value_tf, tf.float32)

    def _np(self, x):
        x1 = x[...,0]
        y = self.value * np.fft.fft2(x1, axes=(-2,-1), norm=self.fftnorm)
        return (y.astype(np.complex64)).reshape(self.shapeOI[0])

    def adj_np(self, y, Type='float'):
        if self.fftnorm=='ortho':
            scaling = 1
        elif self.fftnorm==None:
            scaling = self.shape[0]
        x = scaling * np.fft.ifft2(self.value * y, axes=(-2,-1), norm=self.fftnorm)

        if Type=='float':
            return x.astype(np.float32).reshape(self.shapeOI[1])
        elif Type=='complex':
            return x.astype(np.complex64).reshape(self.shapeOI[1])

    def __call__(self, x):

        x1 = tf.cast(tf.reshape(x, x.shape[:-1]), tf.complex64)
        if self.assign_on_the_fly:
            y1 = self.value_tf.assign(self.value_pl) * tf.compat.v1.fft2d(x1)
        else:
            y1 = self.value_tf * tf.compat.v1.fft2d(x1)

        if self.fftnorm=='ortho':
            return tf.reshape(y1, self.shapeOI[0])/np.sqrt(self.shape[1])
        else:
            return tf.reshape(y1, self.shapeOI[0])

