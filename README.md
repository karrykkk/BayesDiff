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
3. Stable Diffusion
#### Download pre-trained model checkpoint
Download [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt) to `your_local_model_path` 
#### Download data to fit last-layer Laplace (LLLA)
Please download [subset of laion-art](https://drive.google.com/drive/folders/1nL7JQ9bChcCC7LCa3f81kq6LhvRHehYT?usp=drive_link) to `your_local_image_path`. These images is a subset from the [LAION-Art dataset](https://huggingface.co/datasets/laion/laion-art/laion-art.parquet), store it in `your_laion_art_path`. This will allow you to retrieve the corresponding prompts for the downloaded images. Note that a subset of approximately 1000 images is sufficient for effectively fitting the LLLA. 
#### Sample and estimate corresponding pixel-wise uncertainty
In the file `sd.sh`, you will find a template for usage. Please adjust this template to match your local file path and the specific prompt you intend to use.
```shell
cd sd
bash sd.sh
```

## Citation

If you find out work useful, please cite our paper at:

```
@article{
}
```
