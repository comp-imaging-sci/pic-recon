""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import scipy.linalg as la
import scipy.sparse.linalg as sla
try:
    import prox_tv as tv
except ModuleNotFoundError:
    print("ProxTV not available. CAUTION")
import tfwavelets as tfwt

import utils

# Path to test dataset:
# /shared/compton/MRI/varun-glow/mri_data_small.npy
# /shared/compton/MRI/varun-glow/mri_data_test.npy

class ProximalSolver(object):
    """ Reconstruction by simple proximal based regularization.
    """
    # NOTE : Fista can be implemented way more simply in just a few lines in just numpy, 
    # but I have used this to allow for variations, and tf just to keep things consistent
    def __init__(self, sess, fwd, mode=None, data_dtype=tf.float32):
        """ `sess` : Tensorflow session
        `fwd` : Forward model
        """
        self.sess = sess
        self.fwd = fwd
        self.y_meas = tf.compat.v1.placeholder(dtype=self.fwd.output_dtype, name='y_meas')
        self.lamda = tf.compat.v1.placeholder(dtype=tf.float32, name='lamda')
        self.x = tf.compat.v1.placeholder(shape=self.fwd.shapeOI[1], dtype=self.fwd.input_dtype, name='x')
        self.loss = 0.5*tf.norm( self.fwd(self.x) - self.y_meas )**2
        self.loss = tf.real(self.loss)
        self.gradients = tf.compat.v1.gradients(self.loss, self.x)
        self.mode = mode

        # algorithm specific precomputations (fista)
        if mode=='fista':
            self.lip = utils.lipschitz(self.fwd)
    
    def fit(
        self, y_meas,
        step=1.e-3,
        lamda=1.e-3,
        x_init=None,
        n_iter=1000,
        reg_scheme='tv',
        scheduling=False,
        reg_scheduling=False,
        step_schedule_params={'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_param=1.,
        projection_subspace=None,
        verbose=True,
        check_recon_error=False,
        ground_truth=None,
        get_loss_profile=False,
        backtracking=False,
    ):

        if scheduling=='fista':
            print("Using FISTA Scheduling")
            step = step/self.lip

        if backtracking:
            raise NotImplementedError("Fista backtracking not implemented.")

        if get_loss_profile:
            loss_profile = {"iter":[], "loss":[]}
        
        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.sess)

        if not isinstance(x_init, np.ndarray):
            x_init = np.zeros(self.fwd.shapeOI[1], dtype=self.fwd.input_dtype)
        else:
            assert x_init.shape==self.fwd.shapeOI[1], "Shape of z_init does not match shape fo generator input"

        x = x_init.copy()
        y = x_init
        t = 1.
        m = np.zeros(x.shape) # first moment estimate
        v = np.zeros(x.shape) # second moment estimate

        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value}
        else:
            feed_dict = {}

        for i in range(n_iter):
            
            feed_dict.update({
                    self.x: x,
                    self.y_meas: y_meas,
                })
            grad = self.sess.run(self.gradients, feed_dict=feed_dict)[0]
            
            # gradient step
            if not scheduling:
                x = x - step * grad
            if scheduling=='adam':
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad*grad
                mhat = m / (1-beta1)
                vhat = v / (1-beta2)
                x = x - step * mhat / (np.sqrt(vhat) + epsil)
            if scheduling=='simple':
                x = x - step * grad
                step *= beta1
            if scheduling=='increasing':
                step = beta2 - (beta2-epsil)*(beta1)**i
                x = x - step * grad
            if scheduling=='fista':
                xnew = y - step * grad
                if reg_scheme=='tv':
                    xnew[0,:,:] = utils.tv1_2d_cpx(xnew[0,:,:], step*lamda)
                tnew = ( 1 + np.sqrt(1+4*t**2) ) / 2
                y = xnew + (t-1.)/tnew * (xnew - x)
                t = tnew
                x = xnew
                
            # proximal step
            if reg_scheme=='tv' and scheduling!='fista':
                x[0,:,:] = utils.tv1_2d_cpx(x[0,:,:], lamda)

            if reg_scheduling:
                lamda = lamda * reg_schedule_param

            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({self.x: x}) 
                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        # recon_error = self.sess.run(
                        #     la.norm(self.gen(z,Numpy=True)-ground_truth)/la.norm(ground_truth),
                        #     feed_dict={
                        #         self.z: z,
                        #         self.y_meas: y_meas,
                        #         self.lamda: lamda,
                        #     }
                        # )
                        recon_error = la.norm(x-ground_truth)/la.norm(ground_truth),
                        print_string += ", Recon. error : {}".format(recon_error)
                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)

        # if x.dtype == np.float32: x = x*(x>=0.)
        if get_loss_profile:
            return x, loss_profile
        return x


class PriorImageConstrainedSolver1(object):
    """ Regularized gradient descent solver with stylegan
    """

    def __init__(self, 
        gen, 
        fwd, 
        z_regularization_type="l2",
        x_regularization_type=None,
        data_dtype=tf.float32):
        """ 
        `gen` : Generative model
        `fwd` : Forward operator
        """
        self.sess = gen.sess
        self.gen = gen
        self.fwd = fwd
        self.y_meas = tf.compat.v1.placeholder(dtype=data_dtype, name='y_meas')
        self.lamda = tf.compat.v1.placeholder(dtype=tf.float32, name='lamda')
        self.lamda_x = tf.compat.v1.placeholder(dtype=tf.float32, name='lamdax')
        self.z = tf.compat.v1.placeholder(shape=self.gen.shapeOI[1], dtype=tf.float32, name='z')
        self.zn = self.gen.noise_vars
        self.x = self.gen(self.z, self.zn)
        self.err = self.fwd(self.x) - self.y_meas
        self.loss = tf.norm(self.err)**2 / self.gen.shape[0]
        self.z_regularization_type = z_regularization_type
        if z_regularization_type=="l2":
            self.loss = tf.real(self.loss) + self.lamda * tf.reduce_sum([tf.norm(zz)**2 for zz in self.zn]) 
        else:
            self.loss = tf.real(self.loss)

        if x_regularization_type=="tv":
            self.loss += self.lamda_x *tf.image.total_variation(self.x)

        self.gradients = tf.compat.v1.gradients(self.loss, self.zn)


    def fit(
        self, y_meas,
        step=1.e-3,
        lamda=1.e-3,
        zn_init=None,
        prior_image_z=None,
        lamda_x=0,
        n_iter=1000,
        level='noise',
        scheduling=False,
        reg_scheduling=False,
        step_schedule_params={'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_param=0.99,
        verbose=True,
        check_recon_error=False,
        ground_truth=None,
        get_loss_profile=False,
        check_znorm=False,
        ):
        """ Descent iterations.
        Args:
        `y_meas`                : Measured data, np ndarray
        `step`                  : Step size (irrelevant in the case of `increasing` scheduling.)
        `lamda`                 : Regularization parameter (initial)
        `zn_init`               : Noise vector initialization
        `prior_image_z`         : Latent vector of the prior image 
        `lamda_x`               : Parameter for regularization on x
        `n_iter`                : Number of iterations
        `level`                 : The w level at which to impose the prior image constraint.
        `scheduling`            : Step size scheduling
                                 - `None` or `False`: No scheduling 
                                 - `'adam'` : Step size scheduling and momentum updates per adam optimization https://arxiv.org/pdf/1412.6980.pdf
                                 - `'simple'` : update with `step_{i+1} = step_i * beta1`
                                 - `'increasing'` : update with `step_i = beta2 - (beta2-epsil)*(beta1)**i`
        `reg_scheduling`        : Bool. To schedule reg. parameter
        `step_schedule_params`  : Parameters for step size scheduling based on the scheduling mode.
        `reg_schedule_param`    : `lambda = lambda * reg_schedule_param` at each iteration.
        `verbose`               :
        `check_recon_error`     : Whether to calculate and report the reconstruction error. If `True`, `ground_truth` must be provided.
        `ground_truth`          : Ground truth object. use along with `check_recon_error`
        `check_znorm`           : (bool) Compute norm of the current latent estimate.
        """
        if get_loss_profile:
            loss_profile = {"iter": [], "loss": []}

        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        # initialization tweak in the forward model. (If the value of the forward model occupies >2GB memory, pre-initialization in protobuf is not possible, hence on the fly initialization using tf.assign is used)
        # NOTE: Bug: If assign_on_the_fly is True, once the graph is set, another instance of the forward model cannot be created without reseting hte graph.
        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.gen.sess)

        # Make sure that the prior image z has been supplied. Else throw an error
        assert isinstance(prior_image_z, np.ndarray), "Prior image latent space vector must be a numpy array"

        # If initial estimate of latent noise is not given, set it to zero. Else, check its dimensionality
        if not isinstance(zn_init, list):
            zn = np.zeros(self.gen.num_noise_vars, dtype=np.float32)
        else:
            zn = utils.latent_coeffs_to_array(zn_init)
            assert zn.shape[1]==self.gen.num_noise_vars, "Shape of zn_init does not match shape fo generator latent noise"

        m = np.zeros(zn.shape) # first moment estimate
        v = np.zeros(zn.shape) # second moment estimate

        # equate the latent vector to that of the prior image
        z = prior_image_z

        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value}
        else:
            feed_dict = {}

        for i in range(n_iter):

            feed_dict.update({
                self.z: z,
                self.y_meas: y_meas,
                self.lamda: lamda,
                self.lamda_x: lamda_x,
                })
            feed_dict.update({
                var: val for var, val in zip( self.zn, utils.latent_array_to_coeffs(zn, self.gen.dim) )
            })
            grad_full = self.sess.run(
                self.gradients, 
                feed_dict=feed_dict,
            )
            grad = utils.latent_coeffs_to_array(grad_full)
            
            # gradient update for zn
            if not scheduling:
                zn = zn - step * grad
            if scheduling=='adam':
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad*grad
                mhat = m / (1-beta1)
                vhat = v / (1-beta2)
                zn = zn - step * mhat / (np.sqrt(vhat) + epsil)
            if scheduling=='simple':
                zn = zn - step * grad
                step *= beta1
            if scheduling=='increasing':
                step = beta2 - (beta2-epsil)*(beta1)**i
                zn = zn - step * grad

            if self.z_regularization_type=="l1": # soft thresholding on z
                zn = utils.soft(zn, lamda)
        
            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({
                        self.z: z,
                        self.y_meas: y_meas,
                        self.lamda: lamda,
                        self.lamda_x: lamda_x,
                    })
                    feed_dict.update({
                        var: val for var, val in zip( self.zn, utils.latent_array_to_coeffs(zn, self.gen.dim) )
                    })

                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        # recon_error = self.sess.run(
                        #     la.norm(self.gen(z,Numpy=True)-ground_truth)/la.norm(ground_truth),
                        #     feed_dict={
                        #         self.z: z,
                        #         self.y_meas: y_meas,
                        #         self.lamda: lamda,
                        #     }
                        # )
                        recon_error = la.norm(self.gen(z, utils.latent_array_to_coeffs(zn, self.gen.dim), Numpy=True) - ground_truth)/la.norm(ground_truth)
                        print_string += ", Recon. error : {}".format(recon_error)

                    if check_znorm:
                        if self.z_regularization_type=="l2":
                            print_string += ", zn norm : {}".format(la.norm(zn))
                        elif self.z_regularization_type=="l1":
                            print_string += ", zn norm : {}".format(np.sum(abs(zn)))

                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)
        if get_loss_profile:
            return self.gen(z, utils.latent_array_to_coeffs(zn, self.gen.dim), Numpy=True), z, utils.latent_array_to_coeffs(zn, self.gen.dim), loss_profile
            
        return self.gen(z, utils.latent_array_to_coeffs(zn, self.gen.dim), Numpy=True), z, utils.latent_array_to_coeffs(zn, self.gen.dim)


class PriorImageConstrainedSolver2(object):
    """ Regularized gradient descent solver with stylegan
    """

    def __init__(self, 
        gen, 
        fwd, 
        z_regularization_type="l2",
        x_regularization_type=None,
        data_dtype=tf.float32):
        """ 
        `gen` : Generative model
        `fwd` : Forward operator
        """
        self.sess = gen.sess
        self.gen = gen
        self.fwd = fwd
        self.y_meas = tf.compat.v1.placeholder(dtype=data_dtype, name='y_meas')
        self.lamda = tf.compat.v1.placeholder(dtype=tf.float32, name='lamda')
        self.lamda_x = tf.compat.v1.placeholder(dtype=tf.float32, name='lamdax')
        # self.z = tf.compat.v1.placeholder(shape=self.gen.shapeOI[1], dtype=tf.float32, name='z')
        self.w = tf.compat.v1.placeholder(shape=self.gen.shapew, dtype=tf.float32, name='w')
        self.zn = self.gen.noise_vars
        self.x = self.gen(self.w, self.zn, use_latent='w')
        self.err = self.fwd(self.x) - self.y_meas
        self.loss = tf.norm(self.err)**2 / self.gen.shape[0]
        self.z_regularization_type = z_regularization_type
        if z_regularization_type=="l2":
            self.loss = tf.real(self.loss) + self.lamda * tf.reduce_sum([tf.norm(zz)**2 for zz in self.zn]) 
        else:
            self.loss = tf.real(self.loss)

        if x_regularization_type=="tv":
            self.loss += self.lamda_x *tf.image.total_variation(self.x)

        self.gradients = tf.compat.v1.gradients(self.loss, [self.w] + self.zn)

    def fit(
        self, y_meas,
        step=1.e-3,
        lamda=1.e-3,
        z_init=None,
        zn_init=None,
        w_prior_image=None,
        lamda_x=0,
        n_iter=1000,
        levels='noise',
        scheduling=False,
        reg_scheduling=False,
        step_schedule_params={'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_param=0.99,
        verbose=True,
        check_recon_error=False,
        ground_truth=None,
        get_loss_profile=False,
        check_znorm=False,
        ):
        """ Descent iterations.
        Args:
        `y_meas`                : Measured data, np ndarray
        `step`                  : Step size (irrelevant in the case of `increasing` scheduling.)
        `lamda`                 : Regularization parameter (initial)
        `zn_init`               : Noise vector initialization
        `w_prior_image`         : Latent vector of the prior image 
        `lamda_x`               : Parameter for regularization on x
        `n_iter`                : Number of iterations
        `levels`                 : The w levels at which to impose the prior image constraint.
        `scheduling`            : Step size scheduling
                                 - `None` or `False`: No scheduling 
                                 - `'adam'` : Step size scheduling and momentum updates per adam optimization https://arxiv.org/pdf/1412.6980.pdf
                                 - `'simple'` : update with `step_{i+1} = step_i * beta1`
                                 - `'increasing'` : update with `step_i = beta2 - (beta2-epsil)*(beta1)**i`
        `reg_scheduling`        : Bool. To schedule reg. parameter
        `step_schedule_params`  : Parameters for step size scheduling based on the scheduling mode.
        `reg_schedule_param`    : `lambda = lambda * reg_schedule_param` at each iteration.
        `verbose`               :
        `check_recon_error`     : Whether to calculate and report the reconstruction error. If `True`, `ground_truth` must be provided.
        `ground_truth`          : Ground truth object. use along with `check_recon_error`
        `check_znorm`           : (bool) Compute norm of the current latent estimate.
        """
        if get_loss_profile:
            loss_profile = {"iter": [], "loss": []}

        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        # initialization tweak in the forward model. (If the value of the forward model occupies >2GB memory, pre-initialization in protobuf is not possible, hence on the fly initialization using tf.assign is used)
        # NOTE: Bug: If assign_on_the_fly is True, once the graph is set, another instance of the forward model cannot be created without reseting hte graph.
        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.gen.sess)

        # Make sure that the prior image z has been supplied. Else throw an error
        assert isinstance(w_prior_image, np.ndarray), "Prior image latent space vector must be a numpy array"

        # If initial estimate of latent noise is not given, set it to zero. Else, check its dimensionality
        if not isinstance(zn_init, list):
            zn = np.zeros(self.gen.num_noise_vars, dtype=np.float32)
        else:
            zn = utils.latent_coeffs_to_array(zn_init)
            assert zn.shape[1]==self.gen.num_noise_vars, "Shape of zn_init does not match shape fo generator latent noise"

        # If initial estimate of the latent vector is given, estimate a w from it and use a part of it as the initialization (the other part will come from the prior image)
        if not isinstance(z_init, np.ndarray):
            z_init = np.zeros(self.gen.shapeOI[1], dtype=np.float32)
        else:
            assert z_init.shape==self.gen.shapeOI, "Shape of z_init does not match shape fo generator latent vector"

        # Compute the w vector for the prior image. Partially or fully use that.
        # w_prior_image = self.gen.Gs.components.mapping.run(z_prior_image, None)
        if levels == 'noise':
            w = w_prior_image
        else:
            w = self.gen.Gs.components.mapping.run(z_init, None)
            # w[:,levels[0]:levels[1]] = w_prior_image[:,levels[0]:levels[1]]
            w = w_prior_image.copy()
            m = np.zeros(w.shape) # first moment estimate
            v = np.zeros(w.shape) # second moment estimate

        mn = np.zeros(zn.shape) # first moment estimate
        vn = np.zeros(zn.shape) # second moment estimate

        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value}
        else:
            feed_dict = {}

        for i in range(n_iter):

            feed_dict.update({
                self.w: w,
                self.y_meas: y_meas,
                self.lamda: lamda,
                self.lamda_x: lamda_x,
                })
            feed_dict.update({
                var: val for var, val in zip( self.zn, utils.latent_array_to_coeffs(zn, self.gen.dim) )
            })
            grad_full = self.sess.run(
                self.gradients, 
                feed_dict=feed_dict,
            )
            grad_w  = grad_full[0]
            grad_zn = utils.latent_coeffs_to_array(grad_full[1:])

            # gradient update for w
            if levels != 'noise': # dont update w if only optimizing over noise coefficients
                if not scheduling:
                    w[:,levels[0]:levels[1]] = w[:,levels[0]:levels[1]] - step * grad_w[:,levels[0]:levels[1]]
                if scheduling=='adam':
                    m = beta1*m + (1-beta1)*grad_w
                    v = beta2*v + (1-beta2)*grad_w*grad_w
                    mhat = m / (1-beta1)
                    vhat = v / (1-beta2)
                    w[:,levels[0]:levels[1]] = (w - step * mhat / (np.sqrt(vhat) + epsil))[:,levels[0]:levels[1]]
                if scheduling=='simple':
                    w[:,levels[0]:levels[1]] = w[:,levels[0]:levels[1]] - step * grad_w[:,levels[0]:levels[1]]
                    step *= beta1
                if scheduling=='increasing':
                    step = beta2 - (beta2-epsil)*(beta1)**i
                    w[:,levels[0]:levels[1]] = w[:,levels[0]:levels[1]] - step * grad_w[:,levels[0]:levels[1]]

            # gradient update for zn
            if not scheduling:
                zn = zn - step * grad_zn
            if scheduling=='adam':
                mn = beta1*mn + (1-beta1)*grad_zn
                vn = beta2*vn + (1-beta2)*grad_zn*grad_zn
                mhat = mn / (1-beta1)
                vhat = vn / (1-beta2)
                zn = zn - step * mhat / (np.sqrt(vhat) + epsil)
            if scheduling=='simple':
                zn = zn - step * grad_zn
                step *= beta1
            if scheduling=='increasing':
                step = beta2 - (beta2-epsil)*(beta1)**i
                zn = zn - step * grad_zn
        
            # calculate metrics, print stuff
            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({
                        self.w: w,
                        self.y_meas: y_meas,
                        self.lamda: lamda,
                        self.lamda_x: lamda_x,
                    })
                    feed_dict.update({
                        var: val for var, val in zip( self.zn, utils.latent_array_to_coeffs(zn, self.gen.dim) )
                    })

                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        # recon_error = self.sess.run(
                        #     la.norm(self.gen(z,Numpy=True)-ground_truth)/la.norm(ground_truth),
                        #     feed_dict={
                        #         self.z: z,
                        #         self.y_meas: y_meas,
                        #         self.lamda: lamda,
                        #     }
                        # )
                        recon_error = la.norm(self.sess.run(self.x, feed_dict=feed_dict) - ground_truth)/la.norm(ground_truth)
                        print_string += ", Recon. error : {}".format(recon_error)

                    # if check_znorm:
                    #     if self.z_regularization_type=="l2":
                    #         print_string += ", zn norm : {}".format(la.norm(zn))
                    #     elif self.z_regularization_type=="l1":
                    #         print_string += ", zn norm : {}".format(np.sum(abs(zn)))

                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)
        if get_loss_profile:
            return self.gen(w, utils.latent_array_to_coeffs(zn, self.gen.dim), Numpy=True, use_latent='w'), w, utils.latent_array_to_coeffs(zn, self.gen.dim), loss_profile
            
        return self.gen(w, utils.latent_array_to_coeffs(zn, self.gen.dim), Numpy=True, use_latent='w'), w, utils.latent_array_to_coeffs(zn, self.gen.dim)


class PriorImageConstrainedSolver3(object):
    """ Regularized gradient descent solver with stylegan2. Optimizes by keeping all w sections after the breakoff point equal to each other.
    """

    def __init__(self, 
        gen, 
        fwd, 
        optim_varnames=['w', 'zn'],
        regularization_types={'w': None, 'zn': 'l2', 'x': None},
        data_dtype=tf.float32):
        """ 
        `gen`                   : Generative model
        `fwd`                   : Forward operator
        `optim_varnames`        : Names of variables over which to optimize. w optim is partial.
        `regularization_types`  : regularization for various optim vars and the image.
        `data_dtype`            :
        ``
        """
        self.sess = gen.sess
        self.gen = gen
        self.fwd = fwd
        self.optim_varnames = optim_varnames
        print("Optimization variables : ", optim_varnames)
        self.regularization_types = regularization_types

        # Initialize placeholders for the variables
        self.y_meas = tf.compat.v1.placeholder(dtype=data_dtype, name='y_meas')
        self.w  = tf.compat.v1.placeholder(shape=self.gen.shapew, dtype=tf.float32, name='w')
        self.zn = self.gen.noise_vars
        self.x = self.gen(self.w, self.zn, use_latent='w', noise_flattened=True)

        # Get the regularization parameters for the optimization variables
        self.lamdas = {
            nm: tf.compat.v1.placeholder(dtype=tf.float32, name='lamda_'+nm) for nm in optim_varnames + ['x']
            }

        # Get the loss function
        self.err = self.fwd(self.x) - self.y_meas
        self.loss = tf.real(0.5 * tf.norm(self.err)**2)
        for nm in self.optim_varnames + ['x']:
            self.loss += self.lamdas[nm] * self.get_regularizer(
                                                getattr(self, nm), self.regularization_types[nm])

        # get gradients
        self.gradients = tf.compat.v1.gradients(
            self.loss, 
            [self.w] + self.zn,
        )

    def get_regularizer(self, var, regtype):
        if not isinstance(var, list):   var = [var]
        if   regtype == "l2":
            return tf.add_n([tf.norm(v)**2 for v in var])
        elif regtype == "tv":
            return tf.add_n([tf.image.total_variation(v) for v in var])
        elif regtype == None:
            return 0.
        else:
            raise ValueError("Unknown regularization type")

    def fit(
        self, y_meas,
        step                    =   1.e-3,    
        lamdas                  =   {'w': None, 'zn': 1.e-3, 'x': 0.},
        inits                   =   {'w': None, 'zn': None},
        w_prior_image           =   None,
        n_iter                  =   1000,
        cutoff_levels           =   [0,7],
        scheduling              =   'adam',
        reg_scheduling          =   {'w': None, 'zn': None, 'x': None},
        step_schedule_params    =   {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_params     =   {'w': 0.99, 'zn': 0.99, 'x': 0.99},
        verbose                 =   True,
        check_recon_error       =   False,
        ground_truth            =   None,
        get_loss_profile        =   False,
        check_znorm             =   False,
        ):
        """ Descent iterations.
        Args:
        `y_meas`                : Measurements
        `step`                  : Step size
        `lamdas`                : Regularization parameters
        `inits`                 : Initializations
        `w_prior_image`         : W vector of the prior image
        `n_iter`                : Number of iterations
        `cutoff_levels`         : Cutoff level before which w is kept same as the prior image
        `scheduling`            : Use scheduling for step size
        `reg_scheduling`        : Use scheduling for regularization parameter
        `step_schedule_params`  : 
        `reg_schedule_params`   : 
        `verbose`               : Verbose
        `check_recon_error`     : Check recon error with provided ground truth at each iteration
        `ground_truth`          : Must provide if check_recon_error is True
        `get_loss_profile`      : return loss decay profile
        `check_znorm`           : Check norm of the latent vector
        """
        if get_loss_profile:
            loss_profile = {"iter": [], "loss": []}

        # convenience
        l0 = cutoff_levels[0]*2; l1 = cutoff_levels[1]*2

        # set correct scheduling parameters based on the scheduling type
        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        # initialization tweak in the forward model. (If the value of the forward model occupies >2GB memory, pre-initialization in protobuf is not possible, hence on the fly initialization using tf.assign is used)
        # NOTE: Bug: If assign_on_the_fly is True, once the graph is set, another instance of the forward model cannot be created without reseting the graph.
        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.gen.sess)

        # Make sure that the prior image w has been supplied. Else throw an error
        assert isinstance(w_prior_image, np.ndarray), "Prior image latent space vector must be a numpy array"

        # If initial estimate of latent noise is not given, set it to zero. Else, check its dimensionality
        if not isinstance(inits['zn'], np.ndarray):
            zn = np.zeros((1, np.sum([np.prod(self.gen.noise_vars.get_shape().as_list())])), dtype=np.float32)
        else:
            zn = inits['zn']

        # If initial estimate of the w vector is given, use it, otherwise use the prior image w
        if not isinstance(inits['w'], np.ndarray):
            w = w_prior_image.copy()
        else:
            w = inits['w']
            w[:,0:l0] = w_prior_image[:,0:l0]
            w[:,l1:]  = w_prior_image[:,l1:]
            assert list(w.shape)==self.gen.shapew, "Shape of w_init does not match shape fo generator latent vector"

        # First and second moment estimates for adam
        m = np.zeros(w.shape)
        v = np.zeros(w.shape)
        mn = np.zeros(zn.shape)
        vn = np.zeros(zn.shape)

        # Get feeddict with possible initialization tweak in the forward model.
        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value}
        else:
            feed_dict = {}

        for i in range(n_iter):

            feed_dict.update({
                self.w : w,
                self.y_meas : y_meas,
            })
            feed_dict.update({
                var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
            })
            feed_dict.update({
                self.lamdas[nm] : lamdas[nm] for nm in self.optim_varnames + ['x']
            })
            grad_full = self.sess.run(
                self.gradients, 
                feed_dict=feed_dict,
            )

            # get the gradients            
            grad_w  = grad_full[0]
            grad_zn = self.gen.latent_coeffs_to_array(grad_full[1:])

            # gradient updates for w
            if 'w' in self.optim_varnames:
                if not scheduling:
                    w[:,l0:l1] = (w - (l1-l0) * step * grad_w)[:,l0:l1]
                if scheduling=='adam':
                    m = beta1*m + (1-beta1)*grad_w
                    v = beta2*v + (1-beta2)*grad_w*grad_w
                    mhat = m / (1-beta1)
                    vhat = v / (1-beta2)
                    w[:,l0:l1] = (w - (l1-l0) * step * mhat / (np.sqrt(vhat) + epsil) )[:,l0:l1]
                if scheduling=='simple':
                    w[:,l0:l1] = (w - (l1-l0) * step * grad_w)[:, l0:l1]
                    step *= beta1
                if scheduling=='increasing':
                    step = beta2 - (beta2-epsil)*(beta1)**i
                    w[:,l0:l1] = (w - (l1-l0) * step * grad_w)[:, l0:l1]

                # sum the w updates together
                w[:, l0:l1] = np.mean(w[:, l0:l1], axis=1)
            
            # impose prior image constraint
            w[:, 0:l0] = w_prior_image[:, 0:l0]
            w[:, l1:]  = w_prior_image[:, l1:]

            # gradient update for zn
            if 'zn' in self.optim_varnames:
                if not scheduling:
                    zn = zn - step * grad_zn
                if scheduling=='adam':
                    mn = beta1*mn + (1-beta1)*grad_zn
                    vn = beta2*vn + (1-beta2)*grad_zn*grad_zn
                    mhat = mn / (1-beta1)
                    vhat = vn / (1-beta2)
                    zn = zn - step * mhat / (np.sqrt(vhat) + epsil)
                if scheduling=='simple':
                    zn = zn - step * grad_zn
                    step *= beta1
                if scheduling=='increasing':
                    step = beta2 - (beta2-epsil)*(beta1)**i
                    zn = zn - step * grad_zn
        
            # calculate metrics, print stuff
            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({
                        self.w: w,
                        self.y_meas: y_meas,
                    })
                    feed_dict.update({
                        var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
                    })
                    feed_dict.update({
                        self.lamdas[nm] : lamdas[nm] for nm in self.optim_varnames + ['x']
                    })

                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        # recon_error = self.sess.run(
                        #     la.norm(self.gen(z,Numpy=True)-ground_truth)/la.norm(ground_truth),
                        #     feed_dict={
                        #         self.z: z,
                        #         self.y_meas: y_meas,
                        #         self.lamda: lamda,
                        #     }
                        # )
                        recon_error = la.norm(self.sess.run(self.x, feed_dict=feed_dict) - ground_truth)/la.norm(ground_truth)
                        print_string += ", Recon. error : {}".format(recon_error)

                    # if check_znorm:
                    #     if self.z_regularization_type=="l2":
                    #         print_string += ", zn norm : {}".format(la.norm(zn))
                    #     elif self.z_regularization_type=="l1":
                    #         print_string += ", zn norm : {}".format(np.sum(abs(zn)))

                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)

            # Regularization parameter scheduling
            for nm in self.optim_varnames + ['x']:
                if reg_scheduling[nm]:  lamdas[nm] *= reg_schedule_params[nm]

        znest = self.gen.latent_array_to_coeffs(zn)
        if get_loss_profile:
            return self.gen(w, znest, Numpy=True, use_latent='w'), w, znest, loss_profile
            
        return self.gen(w, znest, Numpy=True, use_latent='w'), w, znest


class PriorImageConstrainedSolverNoinv(object):
    """ Regularized gradient descent solver with stylegan2.
    Does not explicitly need the w representation for the prior image.
    """

    def __init__(self, gen, fwd, 
        optim_varnames          = ['w', 'zn'],
        regularization_types    = {'w': None, 'zn': 'l2', 'x': None},
        data_dtype              = tf.float32):
        """ 
        `gen`                   : Generative model
        `fwd`                   : Forward operator
        `optim_varnames`        : Names of variables over which to optimize. w optim is partial.
        `regularization_types`  : regularization for various optim vars and the image.
        `data_dtype`            :
        ``
        """
        self.sess = gen.sess
        self.gen = gen
        self.fwd = fwd
        self.optim_varnames = optim_varnames
        print("Optimization variables : ", optim_varnames)
        self.regularization_types = regularization_types
        
        # Initialize placeholders for the variables
        self.y_meas = tf.compat.v1.placeholder(dtype=data_dtype, name='y_meas')
        self.pi = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, gen.dim, gen.dim, 1], name='pi')
        self.w  = tf.compat.v1.placeholder(shape=self.gen.shapew, dtype=tf.float32, name='w')
        self.wpi = tf.compat.v1.placeholder(shape=self.gen.shapew, dtype=tf.float32, name='wpi')
        self.zn = self.gen.noise_vars
        self.x = self.gen(self.w, self.zn, use_latent='w', noise_flattened=True)
        self.xpi = self.gen(self.wpi, self.zn, use_latent='w', noise_flattened=True)

        # Get the regularization parameters for the optimization variables
        self.alpha = tf.placeholder(dtype=tf.float32, name='alpha') # for prior image loss
        self.lamdas = {
            nm: tf.compat.v1.placeholder(dtype=tf.float32, name='lamda_'+nm) for nm in optim_varnames + ['x']
            }

        # Get the loss function
        self.err = self.fwd(self.x) - self.y_meas
        self.loss = tf.real(0.5 * tf.norm(self.err)**2)

        # Get prior image loss
        self.loss_pi = 0.5 * tf.norm (self.xpi - self.pi)**2
        self.loss += self.alpha * self.loss_pi

        # Get other regularizers
        for nm in self.optim_varnames + ['x']:
            self.loss += self.lamdas[nm] * get_regularizer(
                                                getattr(self, nm), self.regularization_types[nm])

        # get gradients
        self.gradients = tf.compat.v1.gradients(
            self.loss, 
            [self.w, self.wpi] + self.zn,
        )

    def fit(
        self, y_meas,
        step                    =   1.e-03,    
        alpha                   =   1.e+03,
        lamdas                  =   {'w': None, 'zn': 1.e-3, 'x': 0.},
        inits                   =   {'w': None, 'zn': None},
        prior_image             =   None,
        n_iter                  =   1000,
        cutoff_levels           =   [0,7],
        scheduling              =   'adam',
        reg_scheduling          =   {'w': None, 'zn': None, 'x': None, 'wpi': None},
        step_schedule_params    =   {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_params     =   {'w': 0.99, 'zn': 0.99, 'x': 0.99, 'wpi': 1.001},
        verbose                 =   True,
        check_recon_error       =   False,
        ground_truth            =   None,
        get_loss_profile        =   False,
        check_znorm             =   False,
        ):
        """ Descent iterations.
        Args:
        `y_meas`                : Measurements
        `step`                  : Step size
        `alpha`                 : Scalar to control the strength of prior image loss
        `lamdas`                : Regularization parameters
        `inits`                 : Initializations
        `prior_image`           : prior image
        `n_iter`                : Number of iterations
        `cutoff_levels`         : Cutoff level before which w is kept same as the prior image
        `scheduling`            : Use scheduling for step size
        `reg_scheduling`        : Use scheduling for regularization parameter
        `step_schedule_params`  : 
        `reg_schedule_params`   : 
        `verbose`               : Verbose
        `check_recon_error`     : Check recon error with provided ground truth at each iteration
        `ground_truth`          : Must provide if check_recon_error is True
        `get_loss_profile`      : return loss decay profile
        `check_znorm`           : Check norm of the latent vector
        """
        if get_loss_profile:
            loss_profile = {"iter": [], "loss": []}

        # convenience
        l0 = cutoff_levels[0]*2; l1 = cutoff_levels[1]*2

        # set correct scheduling parameters based on the scheduling type
        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        # initialization tweak in the forward model. (If the value of the forward model occupies >2GB memory, pre-initialization in protobuf is not possible, hence on the fly initialization using tf.assign is used)
        # NOTE: Bug: If assign_on_the_fly is True, once the graph is set, another instance of the forward model cannot be created without reseting the graph.
        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.gen.sess)

        # Make sure that the prior image w has been supplied. Else throw an error
        assert isinstance(prior_image, np.ndarray), "Prior image must be a numpy array"

        # If initial estimate of latent noise is not given, set it to zero. Else, check its dimensionality
        if not isinstance(inits['zn'], np.ndarray):
            zn = np.zeros((1, np.sum([np.prod(self.gen.noise_vars.get_shape().as_list())])), dtype=np.float32)
        else:
            zn = inits['zn']

        # If initial estimate of the w vector is given, use it, otherwise use the prior image w
        if not isinstance(inits['w'], np.ndarray):
            w = np.stack([self.gen.wavg.reshape(1,-1)]*self.gen.shapew[1], axis=1) # [1, 14, 512]
            wpi = w.copy()
        else:
            w = inits['w']
            wpi = w.copy()
            assert list(w.shape)==self.gen.shapew, "Shape of w_init does not match shape fo generator latent vector"

        # First and second moment estimates for adam
        m = np.zeros(w.shape)
        v = np.zeros(w.shape)
        mpi = np.zeros(w.shape)
        vpi = np.zeros(w.shape)
        mn = np.zeros(zn.shape)
        vn = np.zeros(zn.shape)

        # Get feeddict with possible initialization tweak in the forward model.
        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value, self.pi: prior_image}
        else:
            feed_dict = {self.pi: prior_image}

        for i in range(n_iter):

            feed_dict.update({
                self.w      : w,
                self.wpi    : wpi,
                self.y_meas : y_meas,
                self.alpha  : alpha,
            })
            feed_dict.update({
                var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
            })
            feed_dict.update({
                self.lamdas[nm] : lamdas[nm] for nm in self.optim_varnames + ['x']
            })
            grad_full = self.sess.run(
                self.gradients, 
                feed_dict=feed_dict,
            )

            # get the gradients            
            grad_w   = grad_full[0]
            grad_wpi = grad_full[1]
            grad_zn  = self.gen.latent_coeffs_to_array(grad_full[2:])

            # gradient updates for w
            if 'w' in self.optim_varnames:
                if (not scheduling) or (scheduling == 'simple'):
                    update_w   = step * grad_w
                    update_wpi = step * grad_wpi

                if scheduling=='adam':
                    m = beta1*m + (1-beta1)*grad_w
                    v = beta2*v + (1-beta2)*grad_w*grad_w
                    mhat = m / (1-beta1); vhat = v / (1-beta2)
                    update_w = step * mhat / (np.sqrt(vhat) + epsil)

                    mpi = beta1*mpi + (1-beta1)*grad_wpi
                    vpi = beta2*vpi + (1-beta2)*grad_wpi*grad_wpi
                    mhat = mpi / (1-beta1); vhat = vpi / (1-beta2)
                    update_wpi = step * mhat / (np.sqrt(vhat) + epsil)

                if scheduling=='simple': step *= beta1

                # combine updates based on the constraints
                update_w    = - np.sum(update_w, axis=1)    # NOTE: PAY ATTENTION TO THE MINUS SIGN
                update_wpi  = - np.sum(update_wpi, axis=1)  # NOTE: PAY ATTENTION TO THE MINUS SIGN
                w[:, :l0]    += update_w[:] + update_wpi[:]
                w[:, l0:l1]  += update_w[:]
                w[:, l1:]    += update_w[:] + update_wpi[:]
                wpi[:,:l0 ]  += update_w[:] + update_wpi[:]
                wpi[:,l0:l1] += update_wpi[:]
                wpi[:,l1: ]  += update_w[:] + update_wpi[:]
            
            # gradient update for zn
            if 'zn' in self.optim_varnames:
                if not scheduling:
                    zn = zn - step * grad_zn
                if scheduling=='adam':
                    mn = beta1*mn + (1-beta1)*grad_zn
                    vn = beta2*vn + (1-beta2)*grad_zn*grad_zn
                    mhat = mn / (1-beta1)
                    vhat = vn / (1-beta2)
                    zn = zn - step * mhat / (np.sqrt(vhat) + epsil)
                if scheduling=='simple':
                    zn = zn - step * grad_zn
                    step *= beta1
                if scheduling=='increasing':
                    step = beta2 - (beta2-epsil)*(beta1)**i
                    zn = zn - step * grad_zn
        
            # calculate metrics, print stuff
            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({
                        self.w      : w,
                        self.wpi    : wpi,
                        self.y_meas : y_meas,
                        self.alpha  : alpha,
                    })
                    feed_dict.update({
                        var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
                    })
                    feed_dict.update({
                        self.lamdas[nm] : lamdas[nm] for nm in self.optim_varnames + ['x']
                    })

                    loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    loss_pi = alpha * self.sess.run(self.loss_pi, feed_dict=feed_dict)
                    print_string = "Iter : {}, Loss : {}".format(i,loss)
                    print_string += ", PIC Loss : {}".format(loss_pi)
                    if check_recon_error:
                        # recon_error = self.sess.run(
                        #     la.norm(self.gen(z,Numpy=True)-ground_truth)/la.norm(ground_truth),
                        #     feed_dict={
                        #         self.z: z,
                        #         self.y_meas: y_meas,
                        #         self.lamda: lamda,
                        #     }
                        # )
                        recon_error = la.norm(self.sess.run(self.x, feed_dict=feed_dict) - ground_truth)/la.norm(ground_truth)
                        print_string += ", Recon. error : {}".format(recon_error)

                    # if check_znorm:
                    #     if self.z_regularization_type=="l2":
                    #         print_string += ", zn norm : {}".format(la.norm(zn))
                    #     elif self.z_regularization_type=="l1":
                    #         print_string += ", zn norm : {}".format(np.sum(abs(zn)))

                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)

            # Regularization parameter scheduling
            for nm in self.optim_varnames + ['x']:
                if reg_scheduling[nm]:  lamdas[nm] *= reg_schedule_params[nm]
            if reg_scheduling['wpi'] :  alpha *= reg_schedule_params['wpi']

        znst = self.gen.latent_array_to_coeffs(zn)
        xest = self.gen(w, znst, Numpy=True, use_latent='w')
        xpi  = self.gen(wpi, znst, Numpy=True, use_latent='w')
        if get_loss_profile:
            return xest, w, xpi, wpi, znst, loss_profile
            
        return xest, w, xpi, wpi, znst


class RegularizedSolver(object):
    """ Regularized gradient descent solver with stylegan
    """

    def __init__(self, 
        gen, 
        fwd, 
        z_regularization_type="l2",
        x_regularization_type=None,
        data_dtype=tf.float32):
        """ 
        `gen` : Generative model
        `fwd` : Forward operator
        """
        self.sess = gen.sess
        self.gen = gen
        self.fwd = fwd
        self.y_meas = tf.compat.v1.placeholder(dtype=data_dtype, name='y_meas')
        self.lamda = tf.compat.v1.placeholder(dtype=tf.float32, name='lamda')
        self.lamda_x = tf.compat.v1.placeholder(dtype=tf.float32, name='lamdax')
        self.z = tf.compat.v1.placeholder(shape=self.gen.shapeOI[1], dtype=tf.float32, name='z')
        self.zn = self.gen.noise_vars
        self.x = self.gen(self.z, self.zn)
        self.err = self.fwd(self.x) - self.y_meas
        self.loss = tf.norm(self.err)**2 / self.gen.shape[0]
        self.z_regularization_type = z_regularization_type
        if z_regularization_type=="l2":
            self.loss = tf.real(self.loss) + self.lamda * tf.norm(self.z)**2 + self.lamda * tf.reduce_sum([tf.norm(zz)**2 for zz in self.zn]) 
        else:
            self.loss = tf.real(self.loss)

        if x_regularization_type=="tv":
            self.loss += self.lamda_x *tf.image.total_variation(self.x)

        self.gradients = tf.compat.v1.gradients(self.loss, [self.z] + self.zn)


    def fit(
        self, y_meas,
        step=1.e-3,
        lamda=1.e-3,
        z_init=None,
        zn_init=None,
        lamda_x=0,
        n_iter=1000,
        scheduling=False,
        reg_scheduling=False,
        step_schedule_params={'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_param=0.99,
        verbose=True,
        check_recon_error=False,
        ground_truth=None,
        get_loss_profile=False,
        check_znorm=False,
        ):
        """ Descent iterations.
        Args:
        `y_meas`                : Measured data, np ndarray
        `step`                  : Step size (irrelevant in the case of `increasing` scheduling.)
        `lamda`                 : Regularization parameter (initial)
        `z_init`                : Latent vector initialization
        `zn_init`               : Latent Noise vector initialization
        `lamda_x`               : Parameter for regularization on x
        `scheduling`            : Step size scheduling
                                 - `None` or `False`: No scheduling 
                                 - `'adam'` : Step size scheduling and momentum updates per adam optimization https://arxiv.org/pdf/1412.6980.pdf
                                 - `'simple'` : update with `step_{i+1} = step_i * beta1`
                                 - `'increasing'` : update with `step_i = beta2 - (beta2-epsil)*(beta1)**i`
        `reg_scheduling`        : Bool. To schedule reg. parameter
        `step_schedule_params`  : Parameters for step size scheduling based on the scheduling mode.
        `reg_schedule_param`    : `lambda = lambda * reg_schedule_param` at each iteration.
        `verbose`               :
        `check_recon_error`     : Whether to calculate and report the reconstruction error. If `True`, `ground_truth` must be provided.
        `ground_truth`          : Ground truth object. use along with `check_recon_error`
        `check_znorm`           : (bool) Compute norm of the current latent estimate.
        """
        if get_loss_profile:
            loss_profile = {"iter": [], "loss": []}

        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        # initialization tweak in the forward model. (If the value of the forward model occupies >2GB memory, pre-initialization in protobuf is not possible, hence on the fly initialization using tf.assign is used)
        # NOTE: Bug: If assign_on_the_fly is True, once the graph is set, another instance of the forward model cannot be created without reseting hte graph.
        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.gen.sess)

        # If the latent vector is not initialized, initialize with a zero vecotr.
        if not isinstance(z_init, np.ndarray):
            z_init = np.zeros(self.gen.shapeOI[1], dtype=np.float32)

        # If initial estimate of latent noise is not given, set it to zero. Else, check its dimensionality
        if not isinstance(zn_init, list):
            zn = np.zeros(self.gen.num_noise_vars, dtype=np.float32)
        else:
            zn = self.gen.latent_coeffs_to_array(zn_init)
            # assert zn.shape[1]==self.gen.num_noise_vars, "Shape of zn_init does not match shape fo generator latent noise"

        z = z_init.copy()

        m = np.zeros(z.shape) # first moment estimate
        v = np.zeros(z.shape) # second moment estimate
        mn = np.zeros(zn.shape) # first moment estimate
        vn = np.zeros(zn.shape) # second moment estimate

        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value}
        else:
            feed_dict = {}

        for i in range(n_iter):

            feed_dict.update({
                self.z: z,
                self.y_meas: y_meas,
                self.lamda: lamda,
                self.lamda_x: lamda_x,
                })
            feed_dict.update({
                var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
            })
            grad_full = self.sess.run(
                self.gradients, 
                feed_dict=feed_dict,
            )
            grad_z = grad_full[0]
            grad = self.gen.latent_coeffs_to_array(grad_full[1:])
            
            # gradient update for z and zn
            if not scheduling:
                z = z - step * grad_z
                zn = zn - step * grad
            if scheduling=='adam':
                mn = beta1*mn + (1-beta1)*grad
                vn = beta2*vn + (1-beta2)*grad*grad
                mhat = mn / (1-beta1)
                vhat = vn / (1-beta2)
                zn = zn - step * mhat / (np.sqrt(vhat) + epsil) #
                m = beta1*m + (1-beta1)*grad_z
                v = beta2*v + (1-beta2)*grad_z*grad_z
                mhat = m / (1-beta1)
                vhat = v / (1-beta2)
                z = z - step * mhat / (np.sqrt(vhat) + epsil)
            if scheduling=='simple':
                zn = zn - step * grad
                z  = z  - step * grad_z
                step *= beta1
            if scheduling=='increasing':
                step = beta2 - (beta2-epsil)*(beta1)**i
                zn = zn - step * grad
                z  = z  - step * grad_z

            if self.z_regularization_type=="l1": # soft thresholding on z
                zn = utils.soft(zn, lamda)
                z  = utils.soft(z , lamda)
        
            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({
                        self.z: z,
                        self.y_meas: y_meas,
                        self.lamda: lamda,
                        self.lamda_x: lamda_x,
                    })
                    feed_dict.update({
                        var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
                    })

                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        # recon_error = self.sess.run(
                        #     la.norm(self.gen(z,Numpy=True)-ground_truth)/la.norm(ground_truth),
                        #     feed_dict={
                        #         self.z: z,
                        #         self.y_meas: y_meas,
                        #         self.lamda: lamda,
                        #     }
                        # )
                        recon_error = la.norm(self.gen(z, self.gen.latent_array_to_coeffs(zn), Numpy=True) - ground_truth)/la.norm(ground_truth)
                        print_string += ", Recon. error : {}".format(recon_error)

                    if check_znorm:
                        if self.z_regularization_type=="l2":
                            print_string += ", z,zn norm : {}".format(la.norm(np.concatenate([z,zn],axis=1)))
                        elif self.z_regularization_type=="l1":
                            print_string += ", z,zn norm : {}".format(np.sum(abs(zn)) + np.sum(abs(z)))

                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)
                    sys.stdout.flush()
        if get_loss_profile:
            return self.gen(z, self.gen.latent_array_to_coeffs(zn), Numpy=True), z, self.gen.latent_array_to_coeffs(zn), loss_profile
            
        return self.gen(z, self.gen.latent_array_to_coeffs(zn), Numpy=True), z, self.gen.latent_array_to_coeffs(zn)


class RegularizedSolverW(object):
    """ Regularized gradient descent solver with stylegan
    """

    def __init__(self, 
        gen, 
        fwd, 
        z_regularization_type="l2",
        x_regularization_type=None,
        data_dtype=tf.float32):
        """ 
        `gen` : Generative model
        `fwd` : Forward operator
        """
        self.sess = gen.sess
        self.gen = gen
        self.fwd = fwd
        self.y_meas = tf.compat.v1.placeholder(dtype=data_dtype, name='y_meas')
        self.lamda = tf.compat.v1.placeholder(dtype=tf.float32, name='lamda')
        self.lamda_x = tf.compat.v1.placeholder(dtype=tf.float32, name='lamdax')
        self.w  = tf.compat.v1.placeholder(shape=self.gen.shapew, dtype=tf.float32, name='w')
        self.zn = self.gen.noise_vars
        self.x = self.gen(self.w, self.zn, use_latent='w')
        self.err = self.fwd(self.x) - self.y_meas
        self.loss = tf.norm(self.err)**2 / self.gen.shape[0]
        self.z_regularization_type = z_regularization_type
        if z_regularization_type=="l2":
            self.loss = tf.real(self.loss) + self.lamda * tf.reduce_sum([tf.norm(zz)**2 for zz in self.zn]) 
        else:
            self.loss = tf.real(self.loss)

        if x_regularization_type=="tv":
            self.loss += self.lamda_x *tf.image.total_variation(self.x)

        self.gradients = tf.compat.v1.gradients(self.loss, [self.w] + self.zn)


    def fit(
        self, y_meas,
        step=1.e-3,
        lamda=1.e-3,
        w_init=None,
        zn_init=None,
        lamda_x=0,
        n_iter=1000,
        scheduling=False,
        reg_scheduling=False,
        step_schedule_params={'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_param=0.99,
        extend=False,
        verbose=True,
        check_recon_error=False,
        ground_truth=None,
        get_loss_profile=False,
        check_znorm=False,
        ):
        """ Descent iterations.
        Args:
        `y_meas`                : Measured data, np ndarray
        `step`                  : Step size (irrelevant in the case of `increasing` scheduling.)
        `lamda`                 : Regularization parameter (initial)
        `z_init`                : Latent vector initialization
        `zn_init`               : Latent Noise vector initialization
        `lamda_x`               : Parameter for regularization on x
        `scheduling`            : Step size scheduling
                                 - `None` or `False`: No scheduling 
                                 - `'adam'` : Step size scheduling and momentum updates per adam optimization https://arxiv.org/pdf/1412.6980.pdf
                                 - `'simple'` : update with `step_{i+1} = step_i * beta1`
                                 - `'increasing'` : update with `step_i = beta2 - (beta2-epsil)*(beta1)**i`
        `reg_scheduling`        : Bool. To schedule reg. parameter
        `step_schedule_params`  : Parameters for step size scheduling based on the scheduling mode.
        `reg_schedule_param`    : `lambda = lambda * reg_schedule_param` at each iteration.
        `verbose`               :
        `check_recon_error`     : Whether to calculate and report the reconstruction error. If `True`, `ground_truth` must be provided.
        `ground_truth`          : Ground truth object. use along with `check_recon_error`
        `check_znorm`           : (bool) Compute norm of the current latent estimate.
        """
        if get_loss_profile:
            loss_profile = {"iter": [], "loss": []}

        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        # initialization tweak in the forward model. (If the value of the forward model occupies >2GB memory, pre-initialization in protobuf is not possible, hence on the fly initialization using tf.assign is used)
        # NOTE: Bug: If assign_on_the_fly is True, once the graph is set, another instance of the forward model cannot be created without reseting hte graph.
        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.gen.sess)

        # If the latent vector is not initialized, initialize with a zero vecotr.
        if not isinstance(w_init, np.ndarray):
            w_init = np.stack([self.gen.wavg.reshape(1,-1)]*self.gen.shapew[1], axis=1) # [1, 14, 512]

        # If initial estimate of latent noise is not given, set it to zero. Else, check its dimensionality
        if not isinstance(zn_init, list):
            zn = np.zeros(self.gen.num_noise_vars, dtype=np.float32)
        else:
            zn = self.gen.latent_coeffs_to_array(zn_init)
            # assert zn.shape[1]==self.gen.num_noise_vars, "Shape of zn_init does not match shape fo generator latent noise"

        w = w_init.copy()

        m = np.zeros(w.shape) # first moment estimate
        v = np.zeros(w.shape) # second moment estimate
        mn = np.zeros(zn.shape) # first moment estimate
        vn = np.zeros(zn.shape) # second moment estimate

        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value}
        else:
            feed_dict = {}

        if not extend:
            step = step*w.shape[1]

        for i in range(n_iter):

            feed_dict.update({
                self.w: w,
                self.y_meas: y_meas,
                self.lamda: lamda,
                self.lamda_x: lamda_x,
                })
            feed_dict.update({
                var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
            })
            grad_full = self.sess.run(
                self.gradients, 
                feed_dict=feed_dict,
            )
            grad_w = grad_full[0]
            grad = self.gen.latent_coeffs_to_array(grad_full[1:])
            
            # gradient update for z and zn
            if not scheduling:
                w = w - step * grad_w
                zn = zn - step * grad
            if scheduling=='adam':
                mn = beta1*mn + (1-beta1)*grad
                vn = beta2*vn + (1-beta2)*grad*grad
                mhat = mn / (1-beta1)
                vhat = vn / (1-beta2)
                zn = zn - step * mhat / (np.sqrt(vhat) + epsil) #
                m = beta1*m + (1-beta1)*grad_w
                v = beta2*v + (1-beta2)*grad_w*grad_w
                mhat = m / (1-beta1)
                vhat = v / (1-beta2)
                w = w - step * mhat / (np.sqrt(vhat) + epsil)
            if scheduling=='simple':
                zn = zn - step * grad
                w  = w  - step * grad_w
                step *= beta1
            if scheduling=='increasing':
                step = beta2 - (beta2-epsil)*(beta1)**i
                zn = zn - step * grad
                w  = w  - step * grad_w

            if not extend:
                w[:,:] = np.mean(w[:,:], axis=1)

            if self.z_regularization_type=="l1": # soft thresholding on z
                zn = utils.soft(zn, lamda)
        
            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({
                        self.w: w,
                        self.y_meas: y_meas,
                        self.lamda: lamda,
                        self.lamda_x: lamda_x,
                    })
                    feed_dict.update({
                        var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
                    })

                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        # recon_error = self.sess.run(
                        #     la.norm(self.gen(z,Numpy=True)-ground_truth)/la.norm(ground_truth),
                        #     feed_dict={
                        #         self.z: z,
                        #         self.y_meas: y_meas,
                        #         self.lamda: lamda,
                        #     }
                        # )
                        recon_error = la.norm(self.gen(w, self.gen.latent_array_to_coeffs(zn), Numpy=True, use_latent='w') - ground_truth)/la.norm(ground_truth)
                        print_string += ", Recon. error : {}".format(recon_error)

                    if check_znorm:
                        if self.z_regularization_type=="l2":
                            print_string += "zn norm : {}".format(la.norm(zn))
                        elif self.z_regularization_type=="l1":
                            print_string += "zn norm : {}".format(np.sum(abs(zn)))

                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)
                    sys.stdout.flush()
        if get_loss_profile:
            return self.gen(w, self.gen.latent_array_to_coeffs(zn), Numpy=True, use_latent='w'), w, self.gen.latent_array_to_coeffs(zn), loss_profile
            
        return self.gen(w, self.gen.latent_array_to_coeffs(zn), Numpy=True, use_latent='w'), w, self.gen.latent_array_to_coeffs(zn)


class RegularizedSolverBase(object):
    """ Common base class for PICGM and CSGM
    """

    def __init__(self, 
        gen, 
        fwd, 
        optim_varnames          = ['w', 'zn'],
        regularization_types    = {'w': None, 'zn': None, 'x': None},
        dim_out                 = None,
        data_dtype              = tf.float32):
        """ 
        `gen`                   : Generative model
        `fwd`                   : Forward operator
        `optim_varnames`        : Names of variables over which to optimize. possible values include w, zn (noise vectors)
        `regularization_types`  : regularization for various optim vars and the image.
        `data_dtype`            :
        ``
        """
        self.sess = gen.sess
        self.gen = gen
        self.fwd = fwd
        self.dim_out = dim_out
        self.optim_varnames = optim_varnames
        print("Optimization variables : ", optim_varnames)
        self.regularization_types = regularization_types

        # Initialize placeholders for the variables
        self.y_meas = tf.compat.v1.placeholder(dtype=data_dtype, name='y_meas')
        self.w  = tf.compat.v1.placeholder(shape=self.gen.shapew, dtype=tf.float32, name='w')
        self.zn = self.gen.noise_vars
        self.x = self.gen(self.w, self.zn, use_latent='w', noise_flattened=True, dim_out=dim_out)
        print(dim_out, self.x.shape)

        # Get the regularization parameters for the optimization variables
        self.lamdas = {
            nm: tf.compat.v1.placeholder(dtype=tf.float32, name='lamda_'+nm) for nm in optim_varnames + ['x']
            }

        # Get the loss function
        self.err = self.fwd(self.x) - self.y_meas
        self.loss = tf.real(0.5 * tf.norm(self.err)**2)
        for nm in self.regularization_types:
            if nm in self.optim_varnames + ['x']:
                self.loss += self.lamdas[nm] * self.get_regularizer(
                                                getattr(self, nm), self.regularization_types[nm])

        # get gradients
        self.gradients = tf.compat.v1.gradients(
            self.loss, 
            [self.w] + self.zn,
        )

    def get_regularizer(self, var, regtype):
        if not isinstance(var, list):   var = [var]
        if   regtype == "l2":
            return tf.add_n([tf.norm(v)**2 for v in var])
        elif regtype == "tv":
            return tf.add_n([tf.image.total_variation(v) for v in var])
        elif regtype == "l2w":
            var = var[0]
            # get covariance matrix
            _,dlatent_samples = self.gen.sample_w(batch_size=10000)
            dlatent_samples = dlatent_samples[:,0:1]
            vlatent_samples = dlatent_samples * ( (dlatent_samples>=0) + 5.*(dlatent_samples<0) )
            vlatent_avg = np.mean(vlatent_samples, axis=0)
            vlatent_cov = np.cov( np.squeeze(vlatent_samples), rowvar=False )
            u,s,v = np.linalg.svd(vlatent_cov)
            vlatent_cov_sqrtinv = ( (u * 1./np.sqrt(s)) @ v ).astype(np.float32)

            # get v vector tensor
            vlatents_var = var * ( 
                tf.cast(var>=0, float) + 5.*tf.cast(var<0, float) ) # undo the last leakyrelu
            nv = tf.tensordot( vlatent_cov_sqrtinv, (vlatents_var - vlatent_avg), [[1],[2]] )
            return tf.norm(nv)**2

        elif regtype == None:
            return 0.
        else:
            raise ValueError("Unknown regularization type")

    def fit(
        self, y_meas,
        step                    =   1.e-3,    
        lamdas                  =   {'w': None, 'zn': 1.e-3, 'x': 0.},
        inits                   =   {'w': None, 'zn': None},
        n_iter                  =   1000,
        scheduling              =   'adam',
        step_schedule_params    =   {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_params     =   {},
        extend                  =   False,
        verbose                 =   True,
        check_recon_error       =   False,
        ground_truth            =   None,
        get_loss_profile        =   False,
        check_znorm             =   False,
        ):
        """ Descent iterations.
        Args:
        `y_meas`                : Measurements
        `step`                  : Step size
        `lamdas`                : Regularization parameters
        `inits`                 : Initializations
        `n_iter`                : Number of iterations
        `scheduling`            : Use scheduling for step size
        `reg_scheduling`        : Use scheduling for regularization parameter
        `step_schedule_params`  : 
        `reg_schedule_params`   : 
        `extend`                : Use extended dlatent space (W+)
        `verbose`               : Verbose
        `check_recon_error`     : Check recon error with provided ground truth at each iteration
        `ground_truth`          : Must provide if check_recon_error is True
        `get_loss_profile`      : return loss decay profile
        `check_znorm`           : Check norm of the latent vector
        """
        if get_loss_profile:
            loss_profile = {"iter": [], "loss": []}

        # set correct scheduling parameters based on the scheduling type
        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        # initialization tweak in the forward model. (If the value of the forward model occupies >2GB memory, pre-initialization in protobuf is not possible, hence on the fly initialization using tf.assign is used)
        # NOTE: Bug: If assign_on_the_fly is True, once the graph is set, another instance of the forward model cannot be created without reseting the graph.
        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.gen.sess)

        # If initial estimate of latent noise is not given, set it to zero. Else, check its dimensionality
        if not isinstance(inits['zn'], np.ndarray):
            zn = np.zeros((1, np.sum([np.prod(self.gen.noise_vars.get_shape().as_list())])), dtype=np.float32)
        else:
            zn = inits['zn']

        # If initial estimate of the w vector is given, use it, otherwise use the prior image w
        if not isinstance(inits['w'], np.ndarray):
            w = np.zeros(self.gen.shapew)
            w[:,:] = self.gen.wavg
        else:
            w = inits['w']
            assert list(w.shape)==self.gen.shapew, "Shape of w_init does not match shape fo generator latent vector"

        # First and second moment estimates for adam
        m = np.zeros(w.shape)
        v = np.zeros(w.shape)
        mn = np.zeros(zn.shape)
        vn = np.zeros(zn.shape)

        # Get feeddict with possible initialization tweak in the forward model.
        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value}
        else:
            feed_dict = {}

        for i in range(n_iter):

            feed_dict.update({
                self.w : w,
                self.y_meas : y_meas,
            })
            feed_dict.update({
                var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
            })
            feed_dict.update({
                self.lamdas[nm] : lamdas[nm] for nm in self.regularization_types
            })
            grad_full = self.sess.run(
                self.gradients, 
                feed_dict=feed_dict,
            )

            # get the gradients            
            grad_w  = grad_full[0]
            grad_zn = self.gen.latent_coeffs_to_array(grad_full[1:])

            # gradient updates for w
            if 'w' in self.optim_varnames:
                if not scheduling:
                    update_w = step * grad_w
                if scheduling=='adam':
                    m = beta1*m + (1-beta1)*grad_w
                    v = beta2*v + (1-beta2)*grad_w*grad_w
                    mhat = m / (1-beta1)
                    vhat = v / (1-beta2)
                    update_w = step * mhat / (np.sqrt(vhat) + epsil)
                if scheduling=='simple':
                    update_w = step * grad_w
                    step *= beta1
                if scheduling=='increasing':
                    step = beta2 - (beta2-epsil)*(beta1)**i
                    update_w = step * grad_w

                # update w using the updates
                update_w *= -1.
                if extend:
                    w += update_w
                else:
                    w = np.sum(update_w, axis=1)
            
            # gradient update for zn
            if 'zn' in self.optim_varnames:
                if not scheduling:
                    zn = zn - step * grad_zn
                if scheduling=='adam':
                    mn = beta1*mn + (1-beta1)*grad_zn
                    vn = beta2*vn + (1-beta2)*grad_zn*grad_zn
                    mhat = mn / (1-beta1)
                    vhat = vn / (1-beta2)
                    zn = zn - step * mhat / (np.sqrt(vhat) + epsil)
                if scheduling=='simple':
                    zn = zn - step * grad_zn
                    step *= beta1
                if scheduling=='increasing':
                    step = beta2 - (beta2-epsil)*(beta1)**i
                    zn = zn - step * grad_zn
        
            # calculate metrics, print stuff
            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({
                        self.w: w,
                        self.y_meas: y_meas,
                    })
                    feed_dict.update({
                        var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
                    })
                    feed_dict.update({
                        self.lamdas[nm] : lamdas[nm] for nm in self.regularization_types
                    })

                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        recon_error = la.norm(self.sess.run(self.x, feed_dict=feed_dict)/2 - ground_truth/2)/np.sqrt(np.prod(ground_truth.shape))
                        print_string += ", Recon. error : {}".format(recon_error)

                    if check_znorm and 'zn' in self.regularization_types:
                        if self.regularization_types['zn']=="l2":
                            print_string += ", zn norm : {}".format(la.norm(zn))

                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)

            # Regularization parameter scheduling
            for nm in self.optim_varnames + ['x']:
                if nm in reg_schedule_params:  lamdas[nm] *= reg_schedule_params[nm]

        znest = self.gen.latent_array_to_coeffs(zn)
        xest  = self.gen(w, znest, Numpy=True, use_latent='w', dim_out=self.dim_out)
        if get_loss_profile:
            return xest, w, znest, loss_profile 

        return xest, w, znest


class PICGMSolver(RegularizedSolverBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(
        self, y_meas,
        step                    =   1.e-3,    
        lamdas                  =   {'w': None, 'zn': 1.e-3, 'x': 0.},
        inits                   =   {'w': None, 'zn': None},
        w_prior_image           =   None,
        n_iter                  =   1000,
        cutoff_levels           =   [0,14],
        scheduling              =   'adam',
        step_schedule_params    =   {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_params     =   {},
        extend                  =   False,
        verbose                 =   True,
        check_recon_error       =   False,
        ground_truth            =   None,
        get_loss_profile        =   False,
        check_znorm             =   False,
        ):
        """ Descent iterations.
        Args:
        `y_meas`                : Measurements
        `step`                  : Step size
        `lamdas`                : Regularization parameters
        `inits`                 : Initializations
        `w_prior_image`         : W vector of the prior image
        `n_iter`                : Number of iterations
        `cutoff_levels`         : Cutoff level before which w is kept same as the prior image
        `scheduling`            : Use scheduling for step size
        `reg_scheduling`        : Use scheduling for regularization parameter
        `step_schedule_params`  : 
        `reg_schedule_params`   : 
        `extend`                : Use extended dlatent space (W+)
        `verbose`               : Verbose
        `check_recon_error`     : Check recon error with provided ground truth at each iteration
        `ground_truth`          : Must provide if check_recon_error is True
        `get_loss_profile`      : return loss decay profile
        `check_znorm`           : Check norm of the latent vector
        """
        if get_loss_profile:
            loss_profile = {"iter": [], "loss": []}

        # convenience
        l0 = cutoff_levels[0]; l1 = cutoff_levels[1]

        # set correct scheduling parameters based on the scheduling type
        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        # initialization tweak in the forward model. (If the value of the forward model occupies >2GB memory, pre-initialization in protobuf is not possible, hence on the fly initialization using tf.assign is used)
        # NOTE: Bug: If assign_on_the_fly is True, once the graph is set, another instance of the forward model cannot be created without reseting the graph.
        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.gen.sess)

        # Make sure that the prior image w has been supplied. Else throw an error
        assert isinstance(w_prior_image, np.ndarray), "Prior image latent space vector must be a numpy array"

        # If initial estimate of latent noise is not given, set it to zero. Else, check its dimensionality
        if not isinstance(inits['zn'], np.ndarray):
            zn = np.zeros((1, np.sum([np.prod(self.gen.noise_vars.get_shape().as_list())])), dtype=np.float32)
        else:
            zn = inits['zn']

        # If initial estimate of the w vector is given, use it, otherwise use the prior image w
        if not isinstance(inits['w'], np.ndarray):
            w = w_prior_image.copy()
        else:
            w = inits['w']
            w[:,0:l0] = w_prior_image[:,0:l0]
            w[:,l1:]  = w_prior_image[:,l1:]
            assert list(w.shape)==self.gen.shapew, "Shape of w_init does not match shape fo generator latent vector"

        # First and second moment estimates for adam
        m = np.zeros(w.shape)
        v = np.zeros(w.shape)
        mn = np.zeros(zn.shape)
        vn = np.zeros(zn.shape)

        # Get feeddict with possible initialization tweak in the forward model.
        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value}
        else:
            feed_dict = {}

        for i in range(n_iter):

            feed_dict.update({
                self.w : w,
                self.y_meas : y_meas,
            })
            feed_dict.update({
                var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
            })
            feed_dict.update({
                self.lamdas[nm] : lamdas[nm] for nm in self.regularization_types
            })
            grad_full = self.sess.run(
                self.gradients, 
                feed_dict=feed_dict,
            )

            # get the gradients            
            grad_w  = grad_full[0]
            grad_zn = self.gen.latent_coeffs_to_array(grad_full[1:])

            # gradient updates for w
            if 'w' in self.optim_varnames:
                if not scheduling:
                    update_w = step * grad_w
                if scheduling=='adam':
                    m = beta1*m + (1-beta1)*grad_w
                    v = beta2*v + (1-beta2)*grad_w*grad_w
                    mhat = m / (1-beta1)
                    vhat = v / (1-beta2)
                    update_w = step * mhat / (np.sqrt(vhat) + epsil)
                if scheduling=='simple':
                    update_w = step * grad_w
                    step *= beta1
                if scheduling=='increasing':
                    step = beta2 - (beta2-epsil)*(beta1)**i
                    update_w = step * grad_w

                # update w using the updates
                update_w *= -1.
                if extend:
                    w[:,l0:l1] += update_w[:,l0:l1]
                else:
                    w[:,l0:l1] = np.sum(update_w[:, l0:l1], axis=1)

            # impose prior image constraint (just in case)
            w[:, 0:l0] = w_prior_image[:, 0:l0]
            w[:, l1:]  = w_prior_image[:, l1:]

            # gradient update for zn
            if 'zn' in self.optim_varnames:
                if not scheduling:
                    zn = zn - step * grad_zn
                if scheduling=='adam':
                    mn = beta1*mn + (1-beta1)*grad_zn
                    vn = beta2*vn + (1-beta2)*grad_zn*grad_zn
                    mhat = mn / (1-beta1)
                    vhat = vn / (1-beta2)
                    zn = zn - step * mhat / (np.sqrt(vhat) + epsil)
                if scheduling=='simple':
                    zn = zn - step * grad_zn
                    step *= beta1
                if scheduling=='increasing':
                    step = beta2 - (beta2-epsil)*(beta1)**i
                    zn = zn - step * grad_zn
        
            # calculate metrics, print stuff
            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({
                        self.w: w,
                        self.y_meas: y_meas,
                    })
                    feed_dict.update({
                        var: val for var, val in zip( self.zn, self.gen.latent_array_to_coeffs(zn) )
                    })
                    feed_dict.update({
                        self.lamdas[nm] : lamdas[nm] for nm in self.regularization_types
                    })

                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        recon_error = la.norm(self.sess.run(self.x, feed_dict=feed_dict)/2 - ground_truth/2)/np.sqrt(np.prod(ground_truth.shape))
                        print_string += ", Recon. error : {}".format(recon_error)

                    if check_znorm and 'zn' in self.regularization_types:
                        if self.regularization_types['zn']=="l2":
                            print_string += ", zn norm : {}".format(la.norm(zn))

                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)

            # Regularization parameter scheduling
            for nm in self.optim_varnames + ['x']:
                if nm in reg_schedule_params:  lamdas[nm] *= reg_schedule_params[nm]

        znest = self.gen.latent_array_to_coeffs(zn)
        if get_loss_profile:
            return self.gen(w, znest, Numpy=True, use_latent='w', dim_out=self.dim_out), w, znest, loss_profile
            
        return self.gen(w, znest, Numpy=True, use_latent='w', dim_out=self.dim_out), w, znest


class PICCSSolver(object):
    """ Reconstruction by simple proximal based regularization.
    """
    # NOTE : Fista can be implemented way more simply in just a few lines in just numpy, 
    # but I have used this to allow for variations, and tf just to keep things consistent
    def __init__(self, sess, fwd, 
        mode                    = None,
        regularization_types    = {'x':'tv', 'pi_err':'l1wt-haar-2'} ,
        weighted_reg            = False,
        data_dtype              = tf.float32):
        """ 
        `sess`                  : Tensorflow session
        `fwd`                   : Forward model
        `regularization_types`  : 'tv' : tv regularization, 'wt': Haar wavelet sparsifying transform
        """
        self.sess = sess
        self.fwd = fwd
        self.y_meas = tf.compat.v1.placeholder(dtype=self.fwd.output_dtype, name='y_meas')
        self.lamda = tf.compat.v1.placeholder(dtype=tf.float32, name='lamda')
        self.alpha = tf.compat.v1.placeholder(dtype=tf.float32, name='alpha')
        self.x = tf.compat.v1.placeholder(shape=self.fwd.shapeOI[1], dtype=self.fwd.input_dtype, name='x')
        self.xpi = tf.compat.v1.placeholder(shape=self.fwd.shapeOI[1], dtype=self.fwd.input_dtype, name='xpi')
        self.pi_err = self.x - self.xpi
        self.loss = 0.5*tf.norm( self.fwd(self.x) - self.y_meas )**2
        self.loss = tf.real(self.loss)
        self.mode = mode
        self.regularization_types = regularization_types
        self.regularization_weights = {}
        self.weighted_reg = weighted_reg

        # algorithm specific precomputations (fista)
        if mode=='fista':
            self.lip = utils.lipschitz(self.fwd)
        if mode=='subgrad':
            self.loss += self.lamda * (1.-self.alpha) * self.get_regularizer(self.x, 'x', regularization_types['x'])
            self.loss += self.lamda * self.alpha * self.get_regularizer(self.pi_err, 'pi_err', regularization_types['pi_err'])

        self.gradients = tf.compat.v1.gradients(self.loss, self.x)

    def get_regularizer(self, var, vartype, regtype):

        if   regtype == "l2":
            return tf.norm(weights * var)**2
        elif regtype == "l1":
            return tf.norm(var, ord=1)
        elif regtype == "tv":
            if self.weighted_reg:
                self.regularization_weights[vartype] = tf.placeholder(dtype=var.dtype, shape=var.shape, name=f'weights_{vartype}')
                return tf.image.total_variation(self.regularization_weights[vartype] * var)
            return tf.image.total_variation(var)
        elif 'l1wt' in regtype:
            _,wt_type,levels = regtype.split('-') # default 'haar', 2
            wavelet = tfwt.parse_wavelet(wt_type)
            coeffs = tfwt.nodes.dwt2d(var[0], wavelet, levels=int(levels))
            if self.weighted_reg:
                self.regularization_weights[vartype] = tf.placeholder(dtype=coeffs.dtype, shape=coeffs.shape, name=f'weights_{vartype}')
                return tf.norm(self.regularization_weights[vartype] * coeffs, ord=1)
            return tf.norm(coeffs, ord=1)
        elif regtype == None:
            return 0.
        else:
            raise ValueError("Unknown regularization type")

    def fit(self, y_meas,
        step                    = 1.e-3,
        lamda                   = 1.e-3,
        alpha                   = 0.5,
        x_init                  = None,
        x_prior_image           = None,
        n_iter                  = 1000,
        scheduling              = False,
        reg_scheduling          = False,
        regularization_weights  = None,
        step_schedule_params    = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8},
        reg_schedule_param      = 1.,
        projection_subspace     = None,
        verbose                 = True,
        check_recon_error       = False,
        ground_truth            = None,
        get_loss_profile        = False,
        backtracking            = False):

        if scheduling=='fista':
            print("Using FISTA Scheduling")
            step = step/self.lip

        if backtracking:
            raise NotImplementedError("Fista backtracking not implemented.")

        if get_loss_profile:
            loss_profile = {"iter":[], "loss":[]}
        
        step_schedule_params0 = {'beta1':0.9, 'beta2':0.999, 'epsil':1.e-8}
        if scheduling=='increasing':
            step_schedule_params0 = {'beta1':0.9, 'beta2':1.e-3, 'epsil':1.e-8}
        for k in list(step_schedule_params.keys()):
            step_schedule_params0[k] = step_schedule_params[k]
        beta1 = step_schedule_params0['beta1']
        beta2 = step_schedule_params0['beta2']
        epsil = step_schedule_params0['epsil']

        if not self.fwd.assign_on_the_fly:
            self.fwd.init.run(session=self.sess)

        if not isinstance(x_init, np.ndarray):
            x_init = np.zeros(self.fwd.shapeOI[1], dtype=self.fwd.input_dtype)
        else:
            assert list(x_init.shape)==self.fwd.shapeOI[1], "Shape of x_init does not match shape of fwd model input"
        
        if not isinstance(x_prior_image, np.ndarray):
            raise ValueError("Must provide prior image")

        x = x_init.copy()
        y = x_init
        t = 1.
        m = np.zeros(x.shape) # first moment estimate
        v = np.zeros(x.shape) # second moment estimate

        if self.fwd.assign_on_the_fly:
            feed_dict = {self.fwd.value_pl: self.fwd.value, self.xpi: x_prior_image}
        else:
            feed_dict = {self.xpi: x_prior_image}

        # update regularization weights
        if self.weighted_reg:
            feed_dict.update({self.regularization_weights[vt] : regularization_weights[vt] for vt in self.regularization_types.keys()})

        for i in range(n_iter):
            
            feed_dict.update({
                    self.x: x,
                    self.y_meas: y_meas,
                    self.lamda: lamda,
                    self.alpha: alpha,
                })
            grad = self.sess.run(self.gradients, feed_dict=feed_dict)[0]
            
            # gradient step
            if not scheduling:
                x = x - step * grad
            if scheduling=='adam':
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad*grad
                mhat = m / (1-beta1)
                vhat = v / (1-beta2)
                x = x - step * mhat / (np.sqrt(vhat) + epsil)
            if scheduling=='simple':
                x = x - step * grad
                step *= beta1
            if scheduling=='increasing':
                step = beta2 - (beta2-epsil)*(beta1)**i
                x = x - step * grad
            if scheduling=='fista':
                xnew = y - step * grad
                if self.regularization_types['x']=='tv':
                    xnew[0,:,:] = utils.tv1_2d_cpx(xnew[0,:,:], step*lamda)
                tnew = ( 1 + np.sqrt(1+4*t**2) ) / 2
                y = xnew + (t-1.)/tnew * (xnew - x)
                t = tnew
                x = xnew
                
            # proximal step
            if self.regularization_types['x']=='tv' and scheduling!='fista' and self.mode != 'subgrad':
                x[0,:,:] = utils.tv1_2d_cpx(x[0,:,:], lamda)

            if reg_scheduling:
                lamda = lamda * reg_schedule_param

            if (i<10) or (i%10==0 and i<100) or (i%50==0) or (i==n_iter-1):
                if verbose:
                    feed_dict.update({self.x: x}) 
                    loss = self.sess.run(self.loss, feed_dict=feed_dict)

                    print_string = "Iter : {}, Loss : {}".format(i, loss)
                    if check_recon_error:
                        recon_error = la.norm(x/2 - ground_truth/2)/np.sqrt(np.prod(ground_truth.shape))
                        print_string += ", Recon. error : {}".format(recon_error)
                    if get_loss_profile:
                        loss_profile["iter"].append(i)
                        loss_profile["loss"].append(loss)

                if verbose:
                    print(print_string)

        # if x.dtype == np.float32: x = x*(x>=0.)
        if get_loss_profile:
            return x, loss_profile
        return x

