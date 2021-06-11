# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
#
# Modified by Varun A. Kelkar - vak2@illinois.edu

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import imageio as io
import os
import glob
from skimage.transform import resize

import projector
import pretrained_networks
from training import dataset
from training import misc

#----------------------------------------------------------------------------

def arange_noise_image(noises):
    noise_img = np.zeros((256,704))
    noise_img[0:64,   0:64]   = np.squeeze(noises[7])
    noise_img[64:128, 0:64]   = np.squeeze(noises[8]) 
    noise_img[128:160,0:32]   = np.squeeze(noises[5])
    noise_img[128:160,32:64]  = np.squeeze(noises[6])
    noise_img[160:176,0:16]   = np.squeeze(noises[3])
    noise_img[160:176,16:32]  = np.squeeze(noises[4])
    noise_img[0:128,  64:192] = np.squeeze(noises[9])
    noise_img[128:256,64:192] = np.squeeze(noises[10])
    noise_img[:,192:448]      = np.squeeze(noises[11])
    noise_img[:,448:]         = np.squeeze(noises[12])
    return noise_img

def project_image(proj, targets, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

#----------------------------------------------------------------------------

def project_image_save_data(proj, targets, origname, projname):
    xorig = targets
    np.save(origname + '.npy', xorig)
    io.imsave(origname + '.png', np.squeeze(np.transpose(xorig, (0,2,3,1))))

    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
    print('\r%-30s\r' % '', end='', flush=True)

    xproj = proj.get_images()
    xproj = resize(xproj, (*xproj.shape[:2], proj.target_dim, proj.target_dim))
    lproj = { 'w': proj.get_dlatents(), 'zn': proj.get_noises() }
    np.save(projname + '.npy', xproj)

    lprojname = os.path.split(projname)
    np.save(lprojname[0] + '/l' + lprojname[1][1:] + '.npy', lproj)
    io.imsave(projname + '.png', np.squeeze(np.transpose(xproj, (0,2,3,1))))

    # noise_img = arange_noise_image(lproj['zn'])
    # io.imsave(lprojname[0] + '/l' + lprojname[1][1:] + '.png', noise_img)

#----------------------------------------------------------------------------

def project_image_style_sensitive(proj, targets, origname, projname):
    xorig = targets
    np.save(origname + '.npy', xorig)
    io.imsave(origname + '.png', np.squeeze(xorig))

    results = proj.run(targets, projname)

    # xproj = results.images
    lproj = { 'w': results.dlatents, 'zn': results.noises }
    # np.save(projname + '.npy', xproj)

    lprojname = os.path.split(projname)
    np.save(lprojname[0] + '/l' + lprojname[1][1:] + '.npy', lproj)
    # io.imsave(projname + '.png', np.squeeze(xproj))

    noise_img = arange_noise_image(lproj['zn'])
    io.imsave(lprojname[0] + '/l' + lprojname[1][1:] + '.png', noise_img)

#----------------------------------------------------------------------------

def project_generated_images(network_pkl, seeds, num_snapshots, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs)
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def project_real_images(network_pkl, dataset_name, data_dir, num_images, num_snapshots):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def project_real_images_from_npy(network_pkl, data_dir, mode, num_images=100):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    if mode == 'normal':
        projector_args = {
            'loss_type'               : {'lpips': 1, 'l2': 0.2},
            'num_steps'               : 5000, 
            'regularization_types'    : ['l2w'],
            'regularize_noise_weight' : {'l2w': 1e-12, 'l2':1e2, 'corr':1e5},
            'optimize_zn'             : False,
            'randomize_noise'         : False,
            'extend_dlatents'         : True,
        }
        print(projector_args)
        proj = projector.Projector(**projector_args)
    elif mode == 'style_sensitive':
        proj = projector.StyleSensitiveProjector(loss_type                  = 'l2',
                                                 num_steps                  = 1000, 
                                                 num_levels                 = 7,
                                                 regularization_types       = ['corr'], 
                                                 regularize_noise_weight    = {'corr':5e5},
                                                 randomize_noise            = False)
    proj.set_network(Gs)

    print('Loading images from "%s"...' % data_dir)
    fnames = sorted(glob.glob(os.path.join(data_dir, 'xpi_*.npy')))
    num_images = min(num_images, len(fnames))

    for i in range(num_images):
        print('Projecting image %d/%d ...' % (i, num_images))
        img = np.fromfile(fnames[i], np.float32).reshape(1,1,256,256)
        img = img - img.min(); img = img / img.max()
        img = 2*img - 1.
        froot = os.path.splitext(os.path.basename(fnames[i]))[0]
        origname = dnnlib.make_run_dir_path(f'xorig_{froot}')
        projname = dnnlib.make_run_dir_path(f'xproj_{froot}')
        if mode == 'normal':
            project_image_save_data(proj, targets=img, origname=origname, projname=projname)
        elif mode == 'style_sensitive':
            project_image_style_sensitive(proj, targets=img, origname=origname, projname=projname)

#----------------------------------------------------------------------------
def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_generated_images_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0)
    project_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    project_real_images_from_npy_parser = subparsers.add_parser('project-real-images-from-npy', help='Project real images from npy')
    project_real_images_from_npy_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_from_npy_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_from_npy_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=2)
    project_real_images_from_npy_parser.add_argument('--mode', type=str, help='Number of images to project (default: %(default)s)', default='normal')
    project_real_images_from_npy_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'project-generated-images'                  : 'run_projector.project_generated_images',
        'project-real-images'                       : 'run_projector.project_real_images',
        'project-real-images-from-npy'              : 'run_projector.project_real_images_from_npy',
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
