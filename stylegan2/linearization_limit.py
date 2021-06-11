""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc

class LinLimitFinder:
    def __init__(self, 
        num_steps                       = 100,
        epsilon                         = 1e-08,
        randomize_noise                 = False,
        optimize_zn                     = False,
        extend_dlatents                 = True,
        ):
        print(f"Epsilon : {epsilon}, optimize_zn : {optimize_zn}")

        self.num_steps                  = num_steps
        self.initial_learning_rate      = 0.001 * epsilon
        self.initial_noise_factor       = 0.0
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.verbose                    = False
        self.clone_net                  = True
        self.randomize_noise            = randomize_noise
        self.optimize_zn                = optimize_zn
        self.extend_dlatents            = extend_dlatents
        self.epsilon                    = epsilon

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._noise_in              = None
        self._dlatents_expr         = None
        self._images_expr           = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None

    def _info(self, *args):
        if self.verbose:
            print('LinLimitFinder:', *args)

    def set_network(self, Gs, minibatch_size=1):
        assert minibatch_size == 1
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Find noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            if self.optimize_zn:
                noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            else:
                noise_init_ops.append(tf.assign(v, tf.zeros(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

        # Prepare target images
        nL, diml = self._Gs.components.synthesis.input_shape[1:]
        self._target_dlatents_var = tf.Variable(tf.zeros(shape=[1,nL,diml]), name='target_dlatents_var')
        self._target_images_var = self._Gs.components.synthesis.get_output_for(self._target_dlatents_var, randomize_noise=self.randomize_noise)


        # Image output graph.
        self._info('Building image output graph...')
        if not self.extend_dlatents:
            self._dlatents_var = tf.Variable(self._target_dlatents_var[:,:1,:], name='dlatents_var')
            self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
            dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._noise_in
            self._dlatents_expr = tf.tile(self._dlatents_var + dlatents_noise, [1, nL, 1])
        else:
            self._dlatents_var = tf.Variable(self._target_dlatents_var, name='dlatents_var')
            self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
            dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._noise_in
            self._dlatents_expr = self._dlatents_var + dlatents_noise

        self._images_expr = self._Gs.components.synthesis.get_output_for(self._dlatents_expr, randomize_noise=self.randomize_noise)

        proc_images_expr = self._images_expr

        # Load jacobian
        self._info("Loading Jacobian...")
        self.J = tf.Variable(tf.zeros([1,256,256,14,512]), name="Jacobian")

        # Loss graph.
        self._info('Building loss graph...')
        delta_f = proc_images_expr - self._target_images_var
        delta_w = self._dlatents_expr - self._target_dlatents_var
        self._dist = tf.reduce_sum(tf.math.abs(
            delta_f[0,0] - tf.tensordot( self.J, delta_w, [ [3,4], [1,2] ] ) 
            )**2)
        self._loss = (-1)*self._dist        # doing gradient ascent

        # dlatent normalization op
        self._info("Getting dlatent normalization")
        self._dlatent_norm_op = tf.assign(
            self._dlatents_var,
            tf.cond(
                tf.norm(delta_w) <= self.epsilon,
                lambda : self._dlatents_var,
                lambda : self._target_dlatents_var + delta_w * self.epsilon * tf.math.rsqrt(tf.reduce_sum(delta_w**2))
            )
        )

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        if (not self.randomize_noise) and self.optimize_zn:
            optim_vars = [self._dlatents_var] + self._noise_vars
        else:
            optim_vars = [self._dlatents_var]
        self._opt.register_gradients(self._loss, optim_vars)
        self._opt_step = self._opt.apply_updates()

    def start(self, target_dlatents, jacobian):
        assert self._Gs is not None

        # Initialize optimization state.
        self._info('Initializing optimization state...')
        nL = self._Gs.components.synthesis.input_shape[1]
        tflib.run(self._noise_init_op)
        tflib.set_vars({
            self._target_dlatents_var   : target_dlatents,
            self.J                      : jacobian,
        })
        tflib.run(self._dlatents_var.initializer)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')

        # Hyperparameters.
        t = self._cur_step / self.num_steps
        noise_strength = self.epsilon * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Train.
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
        tflib.run(self._dlatent_norm_op, feed_dict)

        # Print status.
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            print(dist_value.shape, loss_value)
            self._info('%-8d%-12g%-12g' % (self._cur_step, dist_value, loss_value))
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_expr, {self._noise_in: 0})

    def get_noises(self):
        return tflib.run(self._noise_vars)

    def get_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0})

    def get_loss(self):
        return (-1)*tflib.run(self._loss, {self._noise_in: 0})