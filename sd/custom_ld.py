import torch
import torch.nn as nn
import torch
from laplace.baselaplace import DiagLaplace
from laplace.curvature.backpack import BackPackEF
from torch.nn.utils import parameters_to_vector
import copy

class CustomLD(nn.Module):
    def __init__(self, ld_model, dataloader):

        super().__init__()
        self.unet_feature_extractor = ld_model.model.diffusion_model
        self.conv_out = ld_model.model.diffusion_model.out[2]
        self.copied_conv_out = copy.deepcopy(ld_model.model.diffusion_model.out[2])

        self.unet_feature_extractor.out[2] = nn.Identity()
        
        self.conv_out_la = DiagLaplace(nn.Sequential(self.conv_out, nn.Flatten(1, -1)), likelihood='regression', 
                                    sigma_noise=1, prior_precision=1, prior_mean=0.0, temperature=1.0,
                                    backend=BackPackEF)

        self.fit(dataloader)

    def fit(self, train_loader, override=True):
        
        """Fit the local Laplace approximation at the parameters of the model.
        """
        if override:
            self.conv_out_la._init_H()
            self.conv_out_la.loss = 0
            self.conv_out_la.n_data = 0

        self.conv_out_la.model.eval()
        self.conv_out_la.mean = parameters_to_vector(self.conv_out_la.model.parameters()).detach()

        (z,c,t), _ = next(iter(train_loader))
        with torch.no_grad():
            z = self.unet_feature_extractor(z.to(self.conv_out_la._device), t.to(self.conv_out_la._device), context=c.to(self.conv_out_la._device))
            out = self.conv_out_la.model(z)

        self.conv_out_la.n_outputs = out.shape[-1]
        setattr(self.conv_out_la.model, 'output_size', self.conv_out_la.n_outputs)

        N = len(train_loader.dataset)
        i=0
        for (z, c, t), y in train_loader:
            print(i)
            self.conv_out_la.model.zero_grad()
            z, c, t, y = z.to(self.conv_out_la._device), c.to(self.conv_out_la._device), t.to(self.conv_out_la._device), y.to(self.conv_out_la._device)
            with torch.no_grad():
                z = self.unet_feature_extractor(z, t, context=c)
            loss_batch, H_batch = self.conv_out_la._curv_closure(X=z, y=y, N=N)
            self.conv_out_la.loss += loss_batch
            self.conv_out_la.H += H_batch
            i+=1

        self.conv_out_la.n_data += N
    
    def forward(self, z, t, c):
        
        with torch.no_grad():
            z = self.unet_feature_extractor(z.to(self.conv_out_la._device), t.to(self.conv_out_la._device), context=c.to(self.conv_out_la._device))
        #### nn predict
        f_mean, f_var= self.conv_out_la(z, pred_type='nn', link_approx='mc', n_samples=100)
        f_mean = torch.reshape(f_mean, (-1, 4, 64, 64))
        f_var = torch.reshape(f_var, (-1, 4, 64, 64))
        return (f_mean, f_var)
    
    def accurate_forward(self, z, t, c):

        with torch.no_grad():
            z = self.unet_feature_extractor(z.to(self.conv_out_la._device), t.to(self.conv_out_la._device), context=c.to(self.conv_out_la._device))
            acc_mean = self.copied_conv_out(z)

        return acc_mean