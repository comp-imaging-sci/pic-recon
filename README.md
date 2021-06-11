# Prior image-constrained reconstruction using style-based generative models - Tensorflow implementation

Varun A. Kelkar, Mark A. Anastasio <br />
University of Illinois at Urbana-Champaign, Urbana, IL - 61801, USA

**Paper:** https://arxiv.org/abs/2102.12525

**Contact:** vak2@illinois.edu, maa@illinois.edu

## System Requirements
- Linux/Unix-based systems recommended. The code hasn't been tested on Windows.
- 64 bit Python 3.6+. The code has been tested with Python 3.7.4 installed via Anaconda
- Tensorflow 1.14/1.15. The code has been tested with Tensorflow 1.14. Tensorflow 2+ is not supported.

Additional dependencies that are required for the various reconstruction methods are as follows:
### PLS-TV
- [Prox-TV](https://pythonhosted.org/prox_tv/)
  Can be installed via `pip install prox_tv`. In our experience, this works on Linux and Mac, but not on Windows.
  
### CSGM and PICGM
- Cuda toolkit 10.0 (higher versions may work but haven't been tested)
- GCC 7.2+. The code has been tested with GCC 7.2.0

