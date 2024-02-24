# BayesDiff:  Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference
This repository is our codebase for [BayesDiff](https://arxiv.org/submit/5168441/addfiles).
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
This repository integrates uncertainty quantification into three models, each in its own folder: 

1. ddpm_and_guided - [Guided Diffusion Repository Link](https://github.com/openai/guided-diffusion)
2. sd - [Stable Diffusion Repository Link](https://github.com/CompVis/stable-diffusion)
3. uvit - [U-ViT Repository Link](https://github.com/baofff/U-ViT)

Each folder contains a `custom_model.py` that emerged with uncertainty quantification techniques.

## Usage
### 1. Guided Diffusion
```shell
cd ddpm_and_guided
```
#### Download pre-trained model checkpoint
- Download [imagenet 128x128 ckpt of Guided Diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt) to `your_local_model_path`
- Change the model_path in configs to `your_local_model_path`.
```shell
cd configs
vim imagenet128_guided.yml
```
#### Download data to fit last-layer Laplace (LLLA)
- Please download [Imagenet](https://www.image-net.org/download.php) to `your_local_image_path`.
- Change the `self.image_path` attribute of class imagenet_dataset in `la_train_datasets.py` to `your_local_image_path`.
```shell
vim la_train_datasets.py
```
#### Sample and estimate corresponding pixel-wise uncertainty
In the file `dpm.sh`, you will find a template for usage for UQ-itegrated dpm-solver-2 sampler. By running this bash script, you can get the `sorted_sample.png` based on the image-wise uncertainty metric.
```shell
bash dpm.sh
```
For other samplers, just change `dpm.sh` to `ddpm.sh` or `ddim.sh`.

### 2. Stable Diffusion
```shell
cd sd
```
#### Download pre-trained model checkpoint
Download [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt) to `your_local_model_path` 
#### Download data to fit last-layer Laplace (LLLA)
Please download [subset of laion-art](https://drive.google.com/drive/folders/1nL7JQ9bChcCC7LCa3f81kq6LhvRHehYT?usp=drive_link) to `your_local_image_path`. These images is a subset from the [LAION-Art dataset](https://huggingface.co/camenduru/laion-art/blob/main/laion-art.parquet), store it in `your_laion_art_path`. This will allow you to retrieve the corresponding prompts for the downloaded images. Note that a subset of approximately 1000 images is sufficient for effectively fitting the LLLA. 
#### Sample and estimate corresponding pixel-wise uncertainty
In the file `sd.sh`, you will find a template for usage. Please adjust this template to match your local file path and the specific prompt you intend to use.
```shell
bash sd.sh
```
### 3. U-ViT
```shell
cd uvit
```
#### Download pre-trained model checkpoint
- Download Autoencoder's ckpt from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) which contains image autoencoders converted from Stable Diffusion to `your_local_encoder_path`. Download [ImageNet 256x256 (U-ViT-H/2)](https://drive.google.com/file/d/13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u/view?usp=share_link) to `your_local_uvit_path`.
#### Download data to fit last-layer Laplace (LLLA)
- Please download [Imagenet](https://www.image-net.org/download.php) to `your_local_image_path`.
- Change the `self.image_path` attribute of class imagenet_feature_dataset in `la_train_datasets.py` to `your_local_image_path`.
```shell
vim la_train_datasets.py
```
#### Sample and estimate corresponding pixel-wise uncertainty
In the file `dpm.sh`, you will find a template for usage. Please adjust this template to match your local file path.
```shell
bash dpm.sh
```
## Citation
If you find out work useful, please cite our paper at:

```
@article{
}
```
