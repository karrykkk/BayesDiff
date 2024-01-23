import torch
import torch.nn as nn
import torch
from laplace.baselaplace import DiagLaplace
from laplace.curvature.backpack import BackPackEF
from torch.nn.utils import parameters_to_vector
import copy
    
class CustomUviT(nn.Module):
    def __init__(self, uvit_model, dataloader, args, config):
        super().__init__()
        self.args = args
        self.config = config

        assert self.config.dataset.name == 'imagenet256_features' or self.config.dataset.name == 'imagenet512_features'
        self.decoder_pred = uvit_model.decoder_pred
        self.copied_decoder_pred = copy.deepcopy(uvit_model.decoder_pred)

        self.feature_extractor = uvit_model
        self.feature_extractor.decoder_pred = nn.Identity()
        
        self.decoder_pred_la = DiagLaplace(self.decoder_pred, likelihood='regression', 
                                sigma_noise=1.0, prior_precision=1, prior_mean=0.0, temperature=1.0,
                                backend=BackPackEF)
        self.fit(dataloader)

    def fit(self, train_loader, override=True):
        """Fit the local Laplace approximation at the parameters of the model.
        """
        config = self.config
        assert config.train.mode == 'cond'
        if override:
            self.decoder_pred_la._init_H()
            self.decoder_pred_la.loss = 0
            self.decoder_pred_la.n_data = 0

        self.decoder_pred_la.model.eval()
        self.decoder_pred_la.mean = parameters_to_vector(self.decoder_pred_la.model.parameters()).detach()

        (z,labels,t), _ = next(iter(train_loader))
        model_kwargs = {"y": labels.to(self.decoder_pred_la._device)}
        with torch.no_grad():
            z, L = self.feature_extractor(z.to(self.decoder_pred_la._device), t.to(self.decoder_pred_la._device), **model_kwargs)
            out = self.decoder_pred_la.model(z)
            out = self.feature_extractor.complete_unpatchify(out, L)
        self.decoder_pred_la.n_outputs = out.shape[-1]
        setattr(self.decoder_pred_la.model, 'output_size', self.decoder_pred_la.n_outputs)

        N = len(train_loader.dataset)
        i=0
        for (z, labels, t), y in train_loader:
            print(i)
            self.decoder_pred_la.model.zero_grad()
            
            z, labels, t, y = z.to(self.decoder_pred_la._device), labels.to(self.decoder_pred_la._device), t.to(self.decoder_pred_la._device), y.to(self.decoder_pred_la._device)
            model_kwargs = {"y": labels}
            with torch.no_grad():
                z, L = self.feature_extractor(z, t, **model_kwargs)
            loss_batch, H_batch = self.decoder_pred_la._curv_closure(X=z, L=L, extras = self.feature_extractor.extras, y=y, N=N)
            self.decoder_pred_la.loss += loss_batch
            self.decoder_pred_la.H += H_batch
            i+=1

        self.decoder_pred_la.n_data += N
    
    def forward(self, z, t, **model_kwargs):

        with torch.no_grad():
            z, L = self.feature_extractor(z, t, **model_kwargs)
        #### nn predict
        f_mean, f_var= self.decoder_pred_la(z, pred_type='nn', link_approx='mc', n_samples=100)
        f_mean = self.feature_extractor.complete_unpatchify(f_mean, L)
        f_var = self.feature_extractor.complete_unpatchify(f_var, L)
        
        return (f_mean, f_var)
    
    def accurate_forward(self, z, t, **model_kwargs):

        with torch.no_grad():
            z, L = self.feature_extractor(z, t, **model_kwargs)
            acc_mean = self.copied_decoder_pred(z)
        acc_mean = self.feature_extractor.complete_unpatchify(acc_mean, L)

        return acc_mean