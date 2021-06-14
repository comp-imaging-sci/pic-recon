# StyleGAN2 trained on MR brain and face images: network weights

### Authors: Varun A. Kelkar and Mark A. Anastasio.
University of Illinois at Urbana-Champaign, Urbana, IL - 61801, USA.

The trained weights can be found at `https://databank.illinois.edu/datasets/IDB-4499850`.
The repository contains the network weights for StyleGAN2 [1] models trained on the following data:
1) 200676 T1 and T2 weighted MR brain images from the FastMRI Initiative Database [2] and 866 T1 and T2 weighted MR brain images from the TCIA Brain Tumor Progression dataset [3,4]. The output image size is 256x256. 
    - Name of the weights file: `stylegan2-FastMRIT1T2-config-h.pkl`
    - Real samples from the dataset: `reals-FastMRIT1T2.png`
    - Samples from the model: `fakes-stylegan2-FastMRIT1T2-config-h.png`
2) 164741 T1 and T2 weighted MR brain images from the FastMRI Initiative Database [2], 686 T1 and T2 weighted MR brain images from the TCIA Brain Tumor Progression dataset [3,4], 2206 T1 and T2 weighted images from the TCIA-GBM dataset [4,5], and 36978 T2 weighted images from the OASIS-3 dataset [6]. Output image size 256x256. 
    - Name of the weights file: `stylegan2-CompMRIT1T2-config-f.pkl`
    - Real samples from the dataset: `reals-CompMRIT1T2.png`
    - Samples from the model: `fakes-stylegan2-CompMRIT1T2-config-f.png`
3) 70000 images from the Flickr Faces HQ (FFHQ) dataset [7]. Output image size 128x128x3. 
    - Name of the weights file: `stylegan2-FFHQ-config-f.pkl`
    - Real samples from the dataset: `reals-FFHQ.png`
    - Samples from the model: `fakes-stylegan2-FFHQ-config-f.png`
None of the datasets themselves are included in this repository.

This work was supported in part by NIH Awards EB020604, EB023045, NS102213, EB028652, and NSF Award DMS1614305.

This data is licensed under a Creative Commons Attribution 4.0 International license. (c) Authors.

If using the provided network weights, please also cite the associated work as follows:

BibTeX format:
```
@inproceedings{kelkar2021prior,
  title={Prior Image-Constrained Reconstruction using Style-Based Generative Models},
  author={Kelkar, Varun A and Anastasio, Mark A},
  booktitle={International Conference on Machine Learning},
  pages={},
  year={2021},
  organization={PMLR}
}
```

IEEE format:
```
V. A. Kelkar and M. A. Anastasio, "Prior Image-Constrained Reconstruction using Style-Based Generative Models," in Proceedings of the 38th International Conference on Machine Learning, 2021
```

The preprint of our article can be found at https://arxiv.org/abs/2102.12525. Code related to this dataset and the article can be found at https://github.com/comp-imaging-sci/pic-recon.

References:
1. Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
2. Zbontar, Jure, et al. "fastMRI: An open dataset and benchmarks for accelerated MRI." arXiv preprint arXiv:1811.08839 (2018).
3. Schmainda KM, Prah M (2018). Data from Brain-Tumor-Progression. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2018.15quzvnb 
4. Clark, K., et al., (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057.
5. Scarpace, L., et al. (2016). Radiology Data from The Cancer Genome Atlas Glioblastoma Multiforme [TCGA-GBM] collection [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.RNYFUYE9
6. LaMontagne, Pamela J., et al. "OASIS-3: longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer disease." MedRxiv (2019).
7. Karras, Tero, Samuli Laine, and Timo Aila. "A Style-Based Generator Architecture for Generative Adversarial Networks, 2019 IEEE." CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2018.
