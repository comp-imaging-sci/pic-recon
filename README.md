# Prior image-constrained reconstruction using style-based generative models [ICML 2021] - Tensorflow implementation

### Paper: https://arxiv.org/abs/2102.12525

Varun A. Kelkar, Mark A. Anastasio <br />
University of Illinois at Urbana-Champaign, Urbana, IL - 61801, USA

**Contact:** vak2@illinois.edu, maa@illinois.edu

**Abstract:** Obtaining a useful estimate of an object from highly incomplete imaging measurements remains a holy grail of imaging science. Deep learning methods have shown promise in learning object priors or constraints to improve the conditioning of an ill-posed imaging inverse problem. In this study, a framework for estimating an object of interest that is semantically related to a known prior image, is proposed. An optimization problem is formulated in the disentangled latent space of a style-based generative model, and semantically meaningful constraints are imposed using the disentangled latent representation of the prior image. Stable recovery from incomplete measurements with the help of a prior image is theoretically analyzed. Numerical experiments demonstrating the superior performance of our approach as compared to related methods are presented

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

The simplest way to get a recon algorithm `alg` is as follows:
1. Place the appropriate pretrained StyleGAN2 network `.pkl` file in `stylegan2/nets/`. The current setup uses a StyleGAN2 trained on the [FastMRI initiative database](https://fastmri.med.nyu.edu/) for the MR image study, and one trained on 128x128x3 [FFHQ](https://github.com/NVlabs/ffhq-dataset) images for the face image study. We will provide our pretrained network `.pkl` files soon. Users are free to train their own models and use them. 
2. Enter `pic_recon/scripts/`.
3. Specify the correct network pkl path as `network_path`. (applies only to CSGM and PICGM).
4. Run `bash recon_${alg}.sh` for an algorithm `alg`, where `alg` can be `plstv`, `csgm`, `piccs` or `picgm`.

Further details about the reconstruction are as follows:

The ground truth (GT) images can be found in `pic_recon/ground_truths/`. The prior images (PIs) can be found in `pic_recon/prior_images/`. The GTs and the PIs are organized according to `data_type`, which can be either `faces` or `brain`. For the brain images, we provide two example GT-PI pairs. Additional GT-PI pairs can be downloaded from [The Cancer Imaging Archive (TCIA) Brain-Tumor-Progression dataset](https://wiki.cancerimagingarchive.net/display/Public/Brain-Tumor-Progression#3394811983c589667d0448b7be8e7831cbdcefa6). Links to the data use policy can be found in `pic_recon/ground_truths/brain/README.md`. The Shutterstock Collection containing the GT-PI pairs for the face images can be found here: https://www.shutterstock.com/collections/298591136-e81133c6. A Shutterstock license is needed to use these images. The preprocessing steps used are described in our paper.

All the recon scripts contain the following common arguments:
- `process` : This is the index of the image to be reconstructed.
- `data_type` : Can be either `faces` or `brain`.
- `mask_type` : More generally refers to the forward model. For random Gaussian sensing, this argument takes the form `gaussian_${m_by_n}`. For simulated MRI with random cartesian undersampling, this argument takes the form `mask_rand_${n_by_m}x`. The Gaussian sensing matrices are generated in the code using a fixed seed. The undersampling masks for MRI are located inside `pic_recon/masks_mri/`. 
- `snr` : Measurement SNR.
- `gt_filename`: Path to the ground truth.
- `pi_filename`: (only for PICCS and PICGM) Path to the prior image.
- Various regularization parameters.

The results will be stored in `pic_recon/results/${data_type}/${process}/${algorithm}_${data_type}_${mask_type}_SNR${snr}/`.

## Citations
If you find our code useful, please cite our work as
```
@article{kelkar2021prior,
  title={Prior Image-Constrained Reconstruction using Style-Based Generative Models},
  author={Kelkar, Varun A and Anastasio, Mark A},
  journal={arXiv preprint arXiv:2102.12525},
  year={2021}
}
```








