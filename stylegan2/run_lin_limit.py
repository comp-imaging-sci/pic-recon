""" Copyright (c) 2021, Varun A. Kelkar, Computational Imaging Science Lab @ UIUC.

This work is made available under the MIT License.
Contact: vak2@illinois.edu
"""

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import imageio as io
import os
import glob

import linearization_limit
import pretrained_networks
from training import dataset
from training import misc

#----------------------------------------------------------------------------

def optimize_dlatent_save_data(linlimiter, target_dlatents, jacobian, origname, worsname):

    linlimiter.start(target_dlatents, jacobian)
    loss_profile = []
    while linlimiter.get_cur_step() < linlimiter.num_steps:
        print('\r%d / %d ... ' % (linlimiter.get_cur_step(), linlimiter.num_steps), end='', flush=True)
        linlimiter.step()
        loss_profile.append(linlimiter.get_loss())
    print('\r%-30s\r' % '', end='', flush=True)
    print("Max loss : ", max(loss_profile))

    xwors = linlimiter.get_images()
    lwors = { 'w': linlimiter.get_dlatents(), 'zn': linlimiter.get_noises() }
    np.save(worsname + '.npy', xwors)

    lworsname = os.path.split(worsname)
    np.save(lworsname[0] + '/l' + lworsname[1][1:] + '.npy', lwors)
    io.imsave(worsname + '.png', np.squeeze(xwors))
    return max(loss_profile)

#----------------------------------------------------------------------------

def optimize_linlimit(network_pkl, data_dir, mode, epsilon, num_images=100):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    if mode == 'normal':
        linlimiter_args = {
            "num_steps"                       : 1000,
            "epsilon"                         : epsilon,
            "randomize_noise"                 : False,
            "optimize_zn"                     : False,
            "extend_dlatents"                 : True,
        }
        print(linlimiter_args)
        linlimiter = linearization_limit.LinLimitFinder(**linlimiter_args)

    linlimiter.set_network(Gs)

    print('Loading images from "%s"...' % data_dir)
    wnames = sorted(glob.glob(os.path.join(data_dir, 'w_*.npy')))
    jnames = sorted(glob.glob(os.path.join(data_dir, 'jac*.npy')))
    num_images = min(num_images, len(wnames))
    print(wnames)
    
    for i in range(num_images):
        print('Linmaxing image %d/%d ...' % (i, num_images))
        print(wnames[i], jnames[i])
        wnp = np.load(wnames[i])
        jac = np.load(jnames[i])
        origname = dnnlib.make_run_dir_path(f'xorig_{i}')
        worsname = dnnlib.make_run_dir_path(f'xwors_{i}')
        if mode == 'normal':
            max_loss = optimize_dlatent_save_data(linlimiter, target_dlatents=wnp, jacobian=jac, origname=origname, worsname=worsname)
            print(i, max_loss, file=open('max_losses.txt', 'a'))

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
        description='''StyleGAN2 LinLimiter.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_real_images_from_dat_parser = subparsers.add_parser('optimize-linlimit', help='Maximise the liniear limit')
    project_real_images_from_dat_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_from_dat_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_from_dat_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_from_dat_parser.add_argument('--mode', type=str, help='Number of images to project (default: %(default)s)', default='normal')
    project_real_images_from_dat_parser.add_argument('--epsilon', type=float, help='Epsilon (default: %(default)s)', default=1e-03)
    project_real_images_from_dat_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

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
        'optimize-linlimit'             : 'run_lin_limit.optimize_linlimit',
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
