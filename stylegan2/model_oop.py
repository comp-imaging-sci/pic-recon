""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

import pickle
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

class StyleGAN(object):
    """ Small object-oriented wrapper around StyleGAN2.
    """
    def __init__(self, filename, sess=None):
        """ `filename`  : Path to .pkl file containing the trained weights.
        """
        self.filename = filename
        # self.rnd = np.random.RandomState(seed=sampling_seed)

        if sess==None:
            self.sess = tf.compat.v1.Session().__enter__()
            self.use_destructor = True
        else:
            self.sess = sess
            self.use_destructor = False

        with open(filename, 'rb') as file:
            G, D, Gs = pickle.load(file)
            print('Loaded the pickle file')
        
        self.G = G 
        self.D = D 
        self.Gs = Gs
        self.shapeOI = ( (1, *Gs.output_shape[2:], Gs.output_shape[1]), (1, Gs.input_shape[1]) )
        self.shape = ( np.prod(self.shapeOI[0]), np.prod(self.shapeOI[1]) )
        self.dim = Gs.output_shape[2]

        # get the variables to store the noise vectors
        self.Gsc = Gs.clone()
        self.noise_vars = [var for name, var in self.Gsc.components.synthesis.vars.items() if name.startswith('noise')]
        self.noise_shapes = [var.shape.as_list() for var in self.noise_vars]
        self.num_levels = int( np.ceil( len(self.noise_vars) / 2))

        self.wavg = Gs.get_var('dlatent_avg')
        self.shapew = [1, len(self.noise_vars)+1, Gs.input_shape[1]]


    def __del__(self):

        if self.use_destructor:
            self.sess.close()
            print("Closed session")
        else:
            pass

    def sample(self, temp=1, batch_size=1, seed=None, dim_out=None):

        if seed == None:
            rnd = np.random
        else:
            rnd = np.random.RandomState(seed=seed)
        z = rnd.randn(batch_size, *self.Gs.input_shapes[0][1:]).astype(np.float32)
        noise_vals = []
        noise_vals = [ temp * rnd.randn(*sh).astype(np.float32) for sh in self.noise_shapes]

        w = self.Gs.components.mapping.run(z, None)

        return self.__call__(z, noise_vals, Numpy=True, use_latent='z', dim_out=dim_out), z, w, noise_vals

    def sample_extended(self, temp=1, batch_size=1, seed=None, dim_out=None):
        if seed == None:
            rnd = np.random
        else:
            rnd = np.random.RandomState(seed=seed)
        z = rnd.randn(batch_size, *self.Gs.input_shapes[0][1:]).astype(np.float32)
        noise_vals = []
        noise_vals = [ temp * rnd.randn(*sh).astype(np.float32) for sh in self.noise_shapes]
        w = self.Gs.components.mapping.run(z, None)
        for i in range(1,w.shape[1]):
            z1 = rnd.randn(batch_size, *self.Gs.input_shapes[0][1:]).astype(np.float32)
            w1 = self.Gs.components.mapping.run(z1, None)
            w[:,i] = w1[:,i]

        return self.__call__(w, noise_vals, Numpy=True, use_latent='w', dim_out=dim_out), z, w, noise_vals


    def sample_w(self, temp=1, batch_size=1, seed=None):

        if seed == None:
            rnd = np.random
        else:
            rnd = np.random.RandomState(seed=seed)
        z = temp * rnd.randn(batch_size, *self.Gs.input_shapes[0][1:]).astype(np.float32)
        w = self.Gs.components.mapping.run(z, None)

        return z, w

    # deprecated
    def resample_partial(self, level, z, noise, temp=1, seed=0):
        assert len(noise) == self.num_levels * 2 - 1, "Incorrect length of noise vector"
        assert level < self.num_levels, "Incorrect level specified."
        
        rnd = np.random.RandomState(seed=seed)
        noise = noise.copy()
        for i in range(len(noise)):
            if (i+1) // 2 >= level:
                noise[i]   = temp * rnd.randn(*noise[i]  .shape).astype(np.float32)

        return self.__call__(z, noise, Numpy=True), z.astype(np.float32), noise

    def __call__(self, latent, noise, Numpy=False, use_latent='z', noise_flattened=False, dim_out=None):
        """
        `latent`        : Either the entangled (z) or disentangled (w) latent vector
        `use_latent`    : Is `latent` either `z` or `w`
        """

        bs = latent.shape[0]
        noise_pairs = list(zip(self.noise_vars, noise)) # [(var, val), ...]
        if Numpy:
            tflib.set_vars({var: val for var, val in noise_pairs})

        if Numpy:
            if use_latent == 'z':
                x = self.Gsc.run(latent, None, truncation_psi=1, randomize_noise=False)    
            elif use_latent == 'w':
                x = self.Gsc.components.synthesis.run(latent, is_validation=True, randomize_noise=False)

            if dim_out != None:
                sh = list(x.shape); factor = sh[2] // dim_out
                x = x.reshape(-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor)
                x = np.mean(x, axis=(3,5))

            x = np.transpose(x, [0, 2, 3, 1])
            return x.astype(np.float32)
        else:
            if use_latent == 'z':
                x = self.Gsc.get_output_for(latent, None, truncation_psi=1, randomize_noise=False)
            elif use_latent == 'w':
                x = self.Gsc.components.synthesis.get_output_for(latent, is_validation=True, randomize_noise=False)

            if dim_out != None:
                sh = x.shape.as_list(); factor = sh[2] // dim_out
                x = tf.reduce_mean(tf.reshape(x, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])

            x = tf.transpose(x, [0, 2, 3, 1])
            return tf.cast(x,tf.float32)

    def inv(self, x, Numpy=False):
        raise ValueError("The inverse does not exist")

    def latent_array_to_coeffs(self, zn, Numpy=True):
        if Numpy:   return self.latent_array_to_coeffs_np(zn)
        else:       return self.latent_array_to_coeffs_tf(zn)

    def latent_coeffs_to_array(self, zn, Numpy=True):
        if Numpy:   return self.latent_coeffs_to_array_np(zn)
        else:       return self.latent_coeffs_to_array_tf(zn)        

    def latent_coeffs_to_array_np(self, zn):
        z = [ zz.reshape(zz.shape[0], -1) for zz in zn ]
        z = np.concatenate(z, axis=-1)
        return z

    def latent_coeffs_to_array_tf(self, zn):
        zf = [ tf.reshape(zz, [zz.shape[0], -1]) for zz in zn ]
        zf = tf.concat(zf, axis=-1)
        return zf

    def latent_array_to_coeffs_np(self, zf):
        """
        Changes tensorflow noise inputs from a flattened array to a multiscale bank of noise tensors
        """

        max_dim = self.dim
        sections = [16]
        for i in range(3, int(np.log2(max_dim))+1):
            sections.append(
                sections[-1] + (2**i) **2
            )
            sections.append(
                sections[-1] + (2**i) **2
            )
        sections = sections[:-1]
        zz = np.split(zf, sections, axis=1)
        zz = iter(zz)
        zn = []

        zsingle = next(zz)
        zn.append(
            zsingle.reshape( zsingle.shape[0], 1, 4, 4 )
        )
        for i in range(3, int(np.log2(max_dim))+1):
            zsingle = next(zz)
            zn.append(
                zsingle.reshape( zsingle.shape[0], 1, 2**i, 2**i )
            )
            zsingle = next(zz)
            zn.append(
                zsingle.reshape( zsingle.shape[0], 1, 2**i, 2**i )
            )

        return zn

    def latent_array_to_coeffs_tf(self, zf):
        """
        Changes tensorflow noise inputs from a flattened array to a multiscale bank of noise tensors
        """

        max_dim = self.dim
        sections = []
        for i in range(2, int(np.log2(max_dim))+1):
            sections.append(
                (2**i) **2
            )
            sections.append(
                (2**i) **2
            )
        sections = sections[1:]
        zz = tf.split(zf, sections, axis=1)
        zz = iter(zz)
        zn = []

        zsingle = next(zz)
        zn.append(
            zsingle.reshape( zsingle.shape[0], 1, 4, 4 )
        )
        for i in range(3, int(np.log2(max_dim))+1):
            zsingle = next(zz)
            zn.append(
                zsingle.reshape( zsingle.shape[0], 1, 2**i, 2**i )
            )
            zsingle = next(zz)
            zn.append(
                zsingle.reshape( zsingle.shape[0], 1, 2**i, 2**i )
            )

        return zn


if __name__ == '__main__':

    sess = tf.Session().__enter__()
    filename = 'nets/stylegan2-FastMRIT1T2-config-h.pkl'
    gen = StyleGAN(filename=filename, sess=sess)
    image,z,w,noise = gen.sample()

