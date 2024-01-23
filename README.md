# BayesDiff:  Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference
This repository is our codebase for [arxiv](https://arxiv.org/submit/5168441/addfiles). Our paper is currently under review. We will provide more detailed guide soon.

<p align="center">
  <img width="80%" src="/intro_00.png"/>
</p>

## Installation

```shell
conda create --name BayesDiff python==3.8
conda activate BayesDiff
conda install pip
git clone https://github.com/karrykkk/BayesDiff.git
cd BayesDiff
pip install -r requirements.txt
```
## Framework
This repository integrates uncertainty quantification into three distinct models, each in its own folder: 

1. ddpm_and_guided - [Guided Diffusion Repository Link](https://github.com/openai/guided-diffusion)
2. uvit - [U-ViT Repository Link](https://github.com/baofff/U-ViT)
3. sd - [Stable Diffusion Repository Link](https://github.com/CompVis/stable-diffusion)

Each folder contains a specific model emerged with uncertainty quantification techniques.


## Usage
#### Download pre-trained model checkpoint
#### Download data to fit last-layer Laplace (LLLA)
#### Sample and estimate corresponding pixel-wise uncertainty


## Citation

If you find out work useful, please cite our paper at:

```
@article{
}
```
