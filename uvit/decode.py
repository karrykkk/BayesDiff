import argparse
import os
import torch

from utils import NoiseScheduleVP, stable_diffusion_beta_schedule, amortize
import libs.autoencoder as autoencoder
from custom_uvit import CustomUviT
from libs.uvit import myUViT
from la_train_datasets import imagenet_feature_dataset
import torchvision.utils as tvu
import tqdm
from pytorch_lightning import seed_everything
from utils import inverse_data_transform
from tqdm import tqdm

if __name__ == "__main__":

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    ae = autoencoder.get_model("/home///DiffusionUQ/uvit/assets/stable-diffusion/autoencoder_kl_ema.pth")
    ae.to(device)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return ae.decode(_batch)
    

    def decode_large_batch(_batch):
        decode_mini_batch_size = 1  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs
    
    path = '/data///FID/new_compare/origin/tube_1/dpm_512/ld_sam'
    sam_path = '/data///FID/new_compare/origin/tube_1/dpm_512/sam'
    os.makedirs(sam_path, exist_ok=True)
    img_id = 1000000
    for i in tqdm(range(50000)):
        load = torch.load(os.path.join(path, f'{img_id}.pt'))
        load = torch.unsqueeze(load, dim=0)
        x = inverse_data_transform(decode_large_batch(load))
        tvu.save_image(x, os.path.join(sam_path, f'{img_id}.png'))
        img_id+=1


