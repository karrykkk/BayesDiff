import torch.nn as nn
import torch
from laplace.baselaplace import DiagLaplace
from laplace.curvature.backpack import BackPackEF
from torch.nn.utils import parameters_to_vector
import copy

class CustomModel(nn.Module):
    def __init__(self, diff_model, dataloader, args, config):
        super().__init__()
        self.args = args
        self.config = config

        if self.config.data.dataset == "CELEBA":
            self.conv_out = diff_model.conv_out
            self.copied_cov_out = copy.deepcopy(self.conv_out)

            self.feature_extractor = diff_model
            self.feature_extractor.conv_out = nn.Identity()
            
            self.conv_out_la = DiagLaplace(nn.Sequential(self.conv_out, nn.Flatten(1, -1)), likelihood='regression', 
                                    sigma_noise=1.0, prior_precision=1, prior_mean=0.0, temperature=1.0,
                                    backend=BackPackEF)
            self.fit(dataloader)
        else:
            self.conv_out = diff_model.out[2]
            self.copied_cov_out = copy.deepcopy(self.conv_out)

            self.feature_extractor = diff_model
            self.feature_extractor.out[2] = nn.Identity()

            self.conv_out_la = DiagLaplace(nn.Sequential(self.conv_out, nn.Flatten(1, -1)), likelihood='regression', 
                                    sigma_noise=1, prior_precision=1, prior_mean=0.0, temperature=1.0,
                                    backend=BackPackEF)
            self.fit(dataloader)

    def fit(self, train_loader, override=True):
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
        override : bool, default=True
            whether to initialize H, loss, and n_data again; setting to False is useful for
            online learning settings to accumulate a sequential posterior approximation.
        """
        config = self.config
        if self.config.data.dataset == "CELEBA":
            if override:
                self.conv_out_la._init_H()
                self.conv_out_la.loss = 0
                self.conv_out_la.n_data = 0

            self.conv_out_la.model.eval()
            self.conv_out_la.mean = parameters_to_vector(self.conv_out_la.model.parameters()).detach()

            (X,t), _ = next(iter(train_loader))
            with torch.no_grad():
                try:
                    out = self.conv_out_la.model(self.feature_extractor(X[:1].to(self.conv_out_la._device), t[:1].to(self.conv_out_la._device)))
                except (TypeError, AttributeError):
                    out = self.conv_out_la.model(self.feature_extractor(X.to(self.conv_out_la._device), t.to(self.conv_out_la._device)))
            self.conv_out_la.n_outputs = out.shape[-1]
            setattr(self.conv_out_la.model, 'output_size', self.conv_out_la.n_outputs)

            N = len(train_loader.dataset)
            i=0
            for (X,t), y in train_loader:
                print(i)
                self.conv_out_la.model.zero_grad()
                
                X, t, y = X.to(self.conv_out_la._device), t.to(self.conv_out_la._device), y.to(self.conv_out_la._device)
                with torch.no_grad():
                    X = self.feature_extractor(X, t)
                loss_batch, H_batch = self.conv_out_la._curv_closure(X, y, config, N)
                self.conv_out_la.loss += loss_batch
                self.conv_out_la.H += H_batch
                i+=1

            self.conv_out_la.n_data += N

        else: 
            if override:
                self.conv_out_la._init_H()
                self.conv_out_la.loss = 0
                self.conv_out_la.n_data = 0

            self.conv_out_la.model.eval()
            self.conv_out_la.mean = parameters_to_vector(self.conv_out_la.model.parameters()).detach()

            (X,labels,t), _ = next(iter(train_loader))
            model_kwargs = {"y": labels.to(self.conv_out_la._device)}
            with torch.no_grad():
                out = self.conv_out_la.model(self.feature_extractor(X.to(self.conv_out_la._device), t.to(self.conv_out_la._device), **model_kwargs))
            self.conv_out_la.n_outputs = out.shape[-1]
            setattr(self.conv_out_la.model, 'output_size', self.conv_out_la.n_outputs)

            N = len(train_loader.dataset)
            i=0
            for (X, labels, t), y in train_loader:
                print(i)
                self.conv_out_la.model.zero_grad()
                
                X, labels, t, y = X.to(self.conv_out_la._device), labels.to(self.conv_out_la._device), t.to(self.conv_out_la._device), y.to(self.conv_out_la._device)
                model_kwargs = {"y": labels}
                with torch.no_grad():
                    X = self.feature_extractor(X, t, **model_kwargs)
                loss_batch, H_batch = self.conv_out_la._curv_closure(X, y, config, N)
                self.conv_out_la.loss += loss_batch
                self.conv_out_la.H += H_batch
                i+=1

            self.conv_out_la.n_data += N

    def forward(self, x, t, **model_kwargs):

        if self.config.data.dataset == "CELEBA":
            self.feature_extractor.eval()
            with torch.no_grad():
                x = self.feature_extractor(x, t)
            # #### glm predict
            # f_mean, f_cov = self.conv_out_la(x, pred_type='glm')
            # return f_mean, torch.diagonal(f_cov, dim1=1, dim2=-1)
            #### nn predict
            mean, var = self.conv_out_la(x, pred_type='nn', link_approx='mc', n_samples=100)
            mean = torch.reshape(mean, (-1, 3, self.config.data.image_size, self.config.data.image_size))
            var = torch.reshape(var, (-1, 3, self.config.data.image_size, self.config.data.image_size))
            return (mean, var)

        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                x = self.feature_extractor(x, t, **model_kwargs)
            # #### glm predict
            # f_mean, f_cov= self.conv_out_la(x, pred_type='glm')
            # f_mean = torch.reshape(f_mean, (-1, 6, self.config.data.image_size, self.config.data.image_size))
            # f_var = torch.reshape(torch.diagonal(f_cov,dim1=1, dim2=-1), (-1, 6, self.config.data.image_size, self.config.data.image_size))

            #### nn predict
            f_mean, f_var= self.conv_out_la(x, pred_type='nn', link_approx='mc', n_samples=100)
            f_mean = torch.reshape(f_mean, (-1, 6, self.config.data.image_size, self.config.data.image_size))
            f_var = torch.reshape(f_var, (-1, 6, self.config.data.image_size, self.config.data.image_size))

            f_mean = torch.split(f_mean, 3, dim=1)[0]
            f_var = torch.split(f_var, 3, dim=1)[0]
            return (f_mean, f_var)
    
    def accurate_forward(self, x, t, **model_kwargs):
        
        if self.config.data.dataset == "CELEBA":
            self.feature_extractor.eval()
            with torch.no_grad():
                x = self.feature_extractor(x, t)
                acc_mean = self.copied_cov_out(x)
                
            acc_mean = torch.reshape(acc_mean, (-1, 3, self.config.data.image_size, self.config.data.image_size))
            
            return acc_mean

        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                x = self.feature_extractor(x, t, **model_kwargs)
                acc_mean = self.copied_cov_out(x)

            acc_mean = torch.reshape(acc_mean, (-1, 6, self.config.data.image_size, self.config.data.image_size))
            acc_mean = torch.split(acc_mean, 3, dim=1)[0]          
            
            return acc_mean
