""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

from model_oop import *
import time
import sys
import glob
import os

import imageio as io

image_size = [1,256,256]
stride = 128

filename = 'nets/stylegan2-FastMRIT1T2-config-h.pkl'
sess = tf.Session().__enter__()
gen = StyleGAN(filename=filename, sess=sess)
image,z,w,noise = gen.sample()

noise_pairs = list(zip(gen.noise_vars, noise)) # [(var, val), ...]
tflib.set_vars({var: val*0 for var, val in noise_pairs})

w = tf.Variable(w)
x = gen.Gsc.components.synthesis.get_output_for(w, is_validation=True, randomize_noise=False)

i = tf.placeholder(dtype=tf.int32, shape=[])
k = tf.placeholder(dtype=tf.int32, shape=[])
l = tf.placeholder(dtype=tf.int32, shape=[])
grads = []

for j in range(stride):
    print(j)
    grads.append(tf.gradients(x[0,l,i,k*stride+j], w)[0])

grads = tf.concat(grads, axis=0)

def save_jacobian(jac_path, wnp):

    J = np.zeros((*image_size, wnp.shape[1], 512))
    start = time.time()
    for ll in range(image_size[0]):
        for ii in range(image_size[1]):
            for kk in range(image_size[2]//stride):
                grad = gen.sess.run(grads, feed_dict={l:ll, i:ii, k:kk, w:wnp})
                J[ll,ii,kk*stride:(kk+1)*stride] = grad
    print(time.time()-start)

    np.save(jac_path, J)
    del J


if __name__ == "__main__":

    gpu = int(sys.argv[1])
    sampled = False
    if sampled:
        if not os.path.exists('jacobians'): os.makedirs('jacobians')
        for n in range(50):
            nn = 2*n + gpu

            jac_path = f'jacobians/jac_{nn}.npy'
            _,_,wnp,_ = gen.sample()
            num_levels = wnp.shape[1]

            for i1 in range(1,num_levels):
                _,_,w1,_ = gen.sample()
                wnp[:,i1:] = w1[:,i1:]

            save_jacobian(jac_path, wnp)
            image = gen(wnp, noise, Numpy=True, use_latent='w')
            io.imsave(f'jacobians/img_{nn}.png', np.squeeze(image))
            np.save(f'jacobians/img_{nn}.npy', image)
            np.save(f'jacobians/w_{nn}.npy', wnp)

    else:
        lpaths = glob.glob(sys.argv[2])
        if not os.path.exists('jacobians_expr'): os.makedirs('jacobians_expr')

        for nn, lpath in enumerate(lpaths):
            latent = np.load(lpath, allow_pickle=True).item()
            wnp = latent['w']
            jac_path = f"jacobians_expr/jac{nn}.npy"
            save_jacobian(jac_path, wnp)
            image = gen(wnp, noise, Numpy=True, use_latent='w')
            io.imsave(f'jacobians_expr/img_{nn}.png', np.squeeze(image))
            np.save(f'jacobians_expr/img_{nn}.npy', image)
            np.save(f'jacobians_expr/w_{nn}.npy', wnp)
