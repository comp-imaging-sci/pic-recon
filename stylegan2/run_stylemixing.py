""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

from model_oop import *
import imageio as io
import os
import glob

DIM = 128

def w_mix(gen, levels, worig, wtran, znoise, batch_size=5):
    semilevels = levels
    # for l in levels:
    #     semilevels.append(2*l); semilevels.append(2*l+1)

    wnew = np.stack([worig.copy()]*batch_size, axis=0)
    imnew = []
    for i in range(batch_size):

        im = []
        for j in range(batch_size):
            for s in semilevels:
                wnew[i,j,s,:] = wtran[i,s,:]

        im = gen(wnew[i], znoise, Numpy=True, use_latent='w', dim_out=DIM)
        imnew.append(im)
    return imnew, wnew

def save_mixed_image(im, imorig, imtran, levels, outdir):

    im = np.concatenate(im, axis=1)
    im = np.concatenate(im, axis=1)
    print(im.shape)
    imorig = np.concatenate(imorig, axis=1)
    imtran = np.concatenate([np.zeros(imtran[0].shape), *imtran,], axis=0)

    im = np.concatenate([ imorig, im], axis=0)
    im = np.concatenate([ imtran, im], axis=1)

    io.imsave(f'{outdir}/mix-' + '-'.join([str(l) for l in levels]) + '.png', im)
    return im


if __name__ == '__main__':

    data_type = 'FastMRIT1T2'; config = 'config-f'
    
    sess = tf.Session().__enter__()
    filename = glob.glob(f'stylegan2-{data_type}-{config}.pkl')[0]
    gen = StyleGAN(filename=filename, sess=sess)

    extended = True

    # original and source images
    if not extended:
        imorig, zorig, worig, norig = gen.sample(batch_size=7, temp=1, seed=0, dim_out=DIM)
        imtran, ztran, wtran, ntran = gen.sample(batch_size=7, temp=1, seed=1, dim_out=DIM)
    else:
        imorig, zorig, worig, norig = gen.sample_extended(batch_size=7, temp=0, seed=0, dim_out=DIM)
        imtran, ztran, wtran, ntran = gen.sample_extended(batch_size=7, temp=0, seed=1, dim_out=DIM)

    num_levels = gen.num_levels

    outdir = f'style_mixing/{data_type}-{config}/'
    num_folders = len(glob.glob(outdir+'*'))
    outdir = outdir + f'{num_folders:03}'
    os.makedirs(outdir, exist_ok=True)

    levels = [3,5]
    imnew, wnew = w_mix(gen, levels, worig, wtran, norig, batch_size=7)
    imnew = np.stack(imnew, axis=0)
    save_mixed_image(np.clip(imnew, -1,1), np.clip(imorig,-1,1), np.clip(imtran,-1,1), levels, outdir)
