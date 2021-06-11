""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

from model_oop import *
import os

sess = tf.Session().__enter__()
filename = 'nets/stylegan2-FastMRIT1T2-config-h.pkl'

gen = StyleGAN(filename=filename, sess=sess)
image,z,wnp,noise = gen.sample_extended(batch_size=1)

noise_pairs = list(zip(gen.noise_vars, noise)) # [(var, val), ...]

tflib.set_vars({var: val*0 for var, val in noise_pairs})

w = tf.Variable(wnp)

N = 1
images = gen(w, [n*0 for n in noise], use_latent='w')
ynoise = tf.random_normal([N,256,256,1])
dot = tf.tensordot( images, ynoise, [[1,2,3],[1,2,3]])
jty = tf.gradients(dot, [w])[0]
jtynorm = tf.norm(jty)

if not os.path.exists('jacobians'): os.makedirs('jacobians')
fid = open('jacobians/jacfrobs.txt', 'a')

for i in range(100):
    image,z,wnp,noise = gen.sample_extended(batch_size=1)
    jtynp = np.mean([tflib.run(jtynorm, {w:wnp}) for i in range(100)])
    print(i, jtynp)
    print(i, jtynp, file=fid)


