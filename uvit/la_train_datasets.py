import torch
import numpy as np
import os
import PIL

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps = torch.randn_like(x0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'

class imagenet_feature_dataset(torch.utils.data.Dataset):

    def __init__(self, args, config, ae):

        super().__init__()
        self.image_path = 'your_local_image_path'
        self.paths, self.classes, _ = list_image_files_and_class_recursively(self.image_path)
        self.image_size = config.dataset.image_size
        self.config = config
        self.args = args
        self.betas = stable_diffusion_beta_schedule()
        self.schedule = Schedule(self.betas)
        self.autoencoder = ae

    def __len__(self):
        return len(self.paths)//self.args.train_la_data_size

    def __getitem__(self,idx):
        
        rand_id = idx * torch.randint(low=1, high=self.args.train_la_data_size, size=(1,))[0]
        x_path = self.paths[rand_id]
        label = self.classes[rand_id]
        x = PIL.Image.open(x_path)
        x = x.convert("RGB")       
        x = center_crop_arr(x, self.image_size)
        x = x.astype(np.float32) / (self.image_size/2) - 1 # normalize to [-1, 1]
        x = np.transpose(x, [2, 0, 1])
        x = torch.tensor(x)
        x = torch.unsqueeze(x, dim=0)
        z = self.autoencoder.encode(x.to(f"cuda:{self.args.device}"))
        t, eps, zt = self.schedule.sample(z)
        zt = torch.squeeze(zt, dim=0)
        eps = torch.squeeze(eps, dim=0)
        t = torch.squeeze(t, dim=0)

        return  (zt, label, t), eps

### some utilities #####
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

def list_image_files_and_class_recursively(image_path):
    paths = []
    classes= []
    classes_name = []
    i=0
    for subentry in os.listdir(image_path):
        subfull_path = os.path.join(image_path, subentry)
        for entry in os.listdir(subfull_path):
            full_path = os.path.join(subfull_path, entry)
            paths.append(full_path)
            classes.append(i)
            classes_name.append(subentry)
        i +=1

    return paths, classes, classes_name
