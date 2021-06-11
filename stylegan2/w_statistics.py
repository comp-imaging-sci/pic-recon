""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu

Code to test that the inverse LeakyReLU transformed dlatent vector is indeed approximately multivariate Gaussian.
Based on: 
Wulff, Jonas, and Antonio Torralba. "Improving inversion and generation diversity in stylegan using a gaussianized latent space." arXiv preprint arXiv:2009.06529 (2020).
"""

from model_oop import *
import matplotlib.pyplot as plt

sess = tf.Session().__enter__()

index = 10
config = 'config-h'
filename = f'nets/stylegan2-FastMRIT1T2-{config}.pkl'

gen = StyleGAN(filename=filename, sess=sess)

bs = 50
ws = []
for i in range(10000//bs):
    image,z,w,noise = gen.sample(batch_size=bs)
    ws.append(w[:,0])
    print(i)

wtf = tf.placeholder(dtype=tf.float32, shape=[None, *ws[0].shape[1:]])

# Inverse of the last leaky relu in the mapping function
vtf = wtf * ( tf.cast(wtf>=0, float) + 5.*tf.cast(wtf<0, float) )

vs = []
print("Getting v...")
for i in range(10000//bs):
    v = sess.run(vtf, {wtf: ws[i]})
    print(i)
    vs.append(v)

vs = np.concatenate(vs)
ws = np.concatenate(ws)

# Pick random pairs
i = np.random.randint(ws.shape[-1])
j = np.random.randint(ws.shape[-1])

plt.subplot(211);plt.scatter(ws[:,i], ws[:,j])
plt.subplot(212);plt.scatter(vs[:,i], vs[:,j])
plt.show()

