# Prior image-constrained reconstruction using style-based generative models [ICML 2021] - Tensorflow implementation

### Paper: https://arxiv.org/abs/2102.12525

Varun A. Kelkar, Mark A. Anastasio <br />
University of Illinois at Urbana-Champaign, Urbana, IL - 61801, USA

**Contact:** vak2@illinois.edu, maa@illinois.edu

## System Requirements
- Linux/Unix-based systems recommended. The code hasn't been tested on Windows.
- 64 bit Python 3.6+. The code has been tested with Python 3.7.4 installed via Anaconda
- Tensorflow 1.14/1.15. The code has been tested with Tensorflow 1.14. Tensorflow 2+ is not supported.
- [imageio](https://imageio.readthedocs.io/en/stable/) - Install via `pip` or `conda`.

Additional dependencies that are required for the various reconstruction methods are as follows:
#### PLS-TV
- [Prox-TV](https://pythonhosted.org/prox_tv/)<br />
  Can be installed via `pip install prox_tv`. In our experience, this works on Linux and Mac, but not on Windows.
  
#### CSGM and PICGM
- [Cuda toolkit 10.0](https://developer.nvidia.com/cuda-toolkit) (higher versions may work but haven't been tested)
- GCC 7.2+. The code has been tested with GCC 7.2.0

#### PICCS
- [TF-Wavelets](https://github.com/UiO-CS/tf-wavelets) <br />
  Relevant portions included with this code.
  
## Directory structure
The directory `stylegan2` contains the original StyleGAN2 code, with the following modifications:
- Addition of regularization based on Gaussianized disentangled latent-space for projecting images onto the range of the StyleGAN2 (contained in `stylegan2/projector.py` with regularization and optimization parameters controlled in `stylegan2/run_projector.py`). 
- `stylegan2/model_oop.py` - A small object-oriented wrapper around StyleGAN2.
- Various scripts for related to the theoretical results presented in our paper. 

`stylegan2/nets/` is the location for saving the trained network .pkl files. Stay tuned for our StyleGAN2 network weights, trained on brain MR images and FFHQ dataset images.

The directory `pic_recon` contains the following sub-directories:
- `pic_recon/src` Stores all the scripts for image reconstruction.
- `masks_mri` stores the MRI undersampling masks
- `ground_truths` - images of ground truths
- `prior_images`- Images and latent-projections of prior images.
- `results` - stores the results of the reconstruction.

## Projecting an image onto the latent space of StyleGAN2
1. Make sure `cuda-toolkit/10` and `gcc/7.2+` are loaded.
2. In `run_projector.sh`, set `network` to the path to StyleGAN2 network .pkl. (We will provide our weights soon.) The image size is 256x256x1 for brain images, and 128x128x3 for face images.
3. Run `bash run_projector.sh` from within `stylegan2/`. The projected images along with their latent representations will be stored in `stylegan2/projected_images/`

## Performing image reconstructions:
For all the reconstruction methods, path to ground truth/prior image, regularization parameters, etc need to be set. For CSGM and PICGM, the network path needs to be correctly set. Some examples along with viable values of regularization are given in the scripts `recon_fista.sh`, `recon_piccs.sh`, `recon_csgm.sh`, and `recon_picgm.sh` respectively for the four reconstruction methods, from inside `pic_recon/src`.

Once the parameters are set (most are already set), run (from inside `pic_recon/src`):
- `bash recon_fista.sh` - For PLS-TV
- `bash recon_csgm_w.sh` - For CSGM
- `bash recon_piccs.sh` - For PICCS
- `bash recon_picgm_extended.sh` For PICGM.

If you find our code useful, please cite our work as
```
@article{kelkar2021prior,
  title={Prior Image-Constrained Reconstruction using Style-Based Generative Models},
  author={Kelkar, Varun A and Anastasio, Mark A},
  journal={arXiv preprint arXiv:2102.12525},
  year={2021}
}
```







