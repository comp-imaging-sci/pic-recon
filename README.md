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







