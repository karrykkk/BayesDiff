import os
import numpy as np
import torch
import numpy as np

import PIL
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import pyarrow.parquet as pq
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

class laion_dataset(torch.utils.data.Dataset):

    def __init__(self, model, opt):

        super().__init__()
        table = pq.read_table('/home///DiffusionUQ/sd/data/laion-art/laion-art.parquet')
        self.image_path = '/home///DiffusionUQ/sd/data/laion-art/image_from_url'
        # dirs = os.listdir(self.image_path)
        # print(len(dirs))
        self.entries = os.listdir(self.image_path)
        self.df = table.to_pandas()
        print(self.df.shape[0])
        self.model = model
        self.opt = opt
        self.image_size = 512
        self.num_timesteps = 1000

    def __len__(self):

        return self.opt.train_la_data_size

    def __getitem__(self,idx):
        
        subpath = self.entries[idx]
        x_path = os.path.join(self.image_path, subpath)
        txt = self.df.iloc[int(subpath[:-4]), 1]
        x = Image.open(x_path)
        x = x.convert("RGB")       
        x = center_crop_arr(x, self.image_size)
        x = x.astype(np.float32) / (self.image_size/2) - 1 # normalize to [-1, 1]
        x = np.transpose(x, [2, 0, 1])
        x = torch.tensor(x)
        x = torch.unsqueeze(x, dim=0)
        encoder_posterior = self.model.first_stage_model.encode(x.to(self.model.device))
        z = self.model.get_first_stage_encoding(encoder_posterior).detach()
        c = self.model.get_learned_conditioning([f"{txt}"])
        t = torch.randint(low=0, high=self.num_timesteps, size=(1,))
        e = torch.randn_like(z)
        zt = self.model.q_sample(x_start=z, t=t.to(self.model.device), noise=e.to(self.model.device))
        zt = torch.squeeze(zt, dim=0)
        e = torch.squeeze(e, dim=0)
        t = torch.squeeze(t, dim=0)
        c = torch.squeeze(c, dim=0)

        return  (zt, c, t), torch.flatten(e, start_dim=0, end_dim=-1)
        
