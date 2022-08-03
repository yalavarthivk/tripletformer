# pylint: disable=E1101
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import utils_prop_att
from layers import UnTAN
import pdb

import properscoring as ps
from scipy.stats import norm


class Gaussian:
    mean = None
    logvar = None

class LossInfo:
    px = None
    loglik = None
    elbo = None
    kl = None
    mse = None
    mae = None
    mean_mse = None
    mean_mae = None
    mogloglik = None
    composite_loss = None

class GPR(nn.Module):
    def __init__(self, l=True, sigma_f=True, sigma_y=True, device='cuda'):
        super(GPR, self).__init__()
        if l:
            self.l = nn.Parameter(torch.randn(1))
        else:
            self.register_parameter('l', None)
        if sigma_f:
            self.sigma_f = nn.Parameter(torch.randn(1))
        else:
            self.register_parameter('sigma_f', None)
        if sigma_y:
            self.sigma_y = nn.Parameter(torch.randn(1))
        else:
            self.register_parameter('sigma_y', None)
        self.device = device
        self.relu = nn.ReLU()
        # self.reset_parameters()
    
    def reset_parameters(self):
        self.l.data.fill_(1)
        self.sigma_f.data.fill(1)
    def cov(self, x, x_str):
        # pdb.set_trace()
        x = x[:,:,None,:].repeat(1,1, x_str.shape[1],1)
        x_str = x_str[:,None,:,:].repeat(1, x.shape[1],1,1)
        dif = (x - x_str)**2
        dif = dif.sum(dim=-1)
        dif = dif**0.5
        K = (self.sigma_f**2)*(torch.exp(-dif/(2*(self.l**2))))
        # if K.isnan().any():
        #     pdb.set_trace()
        return K
    def forward(self, X,y,c_m,X_str,t_m):
        # pdb.set_trace()
        X = X*c_m[:,:,None]
        X_str = X_str*t_m[:,:,None]

        #comupute mu
        div_fact = torch.sum(c_m[:,:,None], dim=-1, keepdim=True)
        div_fact[div_fact==0] = 1
        mu = torch.sum(X, dim=-1, keepdim=True)/div_fact
        # mu = mu.repeat(1,X.size(1),1)
        mu *= c_m[:,:,None]

        #compute mu_str
        div_fact_str = torch.sum(t_m[:,:,None], dim=-1, keepdim=True)
        div_fact_str[div_fact_str==0] = 1
        mu_str = torch.sum(X_str, dim=-1, keepdim=True)/div_fact_str
        # mu_str = mu_str.repeat(1,X_str.size(1),1)
        mu_str *= t_m[:,:,None]


        #Compute Sigma
        Sigma = self.cov(X,X)
        Sigma_eye = torch.eye(Sigma.shape[-1]).unsqueeze(0).repeat(Sigma.shape[0],1,1).to(X.device)
        eye = torch.eye(Sigma.shape[-1]).unsqueeze(0).repeat(Sigma.shape[0],1,1).to(X.device) * self.sigma_y**2
        Sigma += eye
        Sigma_mask = torch.matmul(c_m[:,:,None], c_m[:,None,:])
        Sigma *= Sigma_mask
        Sigma_eye *= 1-Sigma_mask
        Sigma += Sigma_eye

        #Compute Simga_str
        
        Sigma_str = self.cov(X,X_str)
        Sigma_str_mask = torch.matmul(c_m[:,:,None], t_m[:,None,:])
        Sigma_str = Sigma_str*Sigma_str_mask


        #Compute Simga_str_str
        Sigma_str_str = self.cov(X_str, X_str)
        Sigma_str_str_mask = torch.matmul(t_m[:,:,None], t_m[:,None,:])
        Sigma_str_str = Sigma_str_str*Sigma_str_str_mask

        Sigma_inv = Sigma.inverse()
        temp_mean = y[:,:,None]-mu
        temp_mean = temp_mean*c_m[:,:,None]
        
        # mu_pred = mu_str + torch.matmul(Sigma_str.transpose(-1,-2), torch.matmul(Sigma_inv, temp_mean))
        mu_pred = torch.matmul(Sigma_str.transpose(-1,-2), torch.matmul(Sigma_inv, y[:,:,None]))

        eye_str_str = torch.eye(Sigma_str_str.shape[-1]).unsqueeze(0).repeat(Sigma.shape[0],1,1).to(X.device)
        Sigma_pred = Sigma_str_str + eye_str_str*self.sigma_y**2 - torch.matmul(Sigma_str.transpose(-1,-2), torch.matmul(Sigma_inv, Sigma_str))
        Sigma_pred = self.relu(Sigma_pred)
        return mu_pred, Sigma_pred.diagonal(dim1=-2, dim2=-1)[:,:,None]


class GP(nn.Module):
    '''Temporal Variational Autoencoder'''
    def __init__(
        self,
        input_dim,
        device='cuda',
        norm=True
    ):
        super().__init__()
        self.dim = input_dim
        self.device = device
        # self.gpr = GPR()
        self.norm=norm
        self.gpr = nn.ModuleList()
        for i in range(input_dim):
            self.gpr.append(GPR(device=device).to(device))


    def compute_logvar(self, sigma):
        if self.is_constant:
            sigma = torch.zeros(sigma.size()) + self.std
        elif self.is_bounded:
            sigma = 0.01 + F.softplus(sigma)
        elif self.is_constant_per_dim:
            sigma = 0.01 + F.softplus(sigma)
        else:
            return sigma
        return 2 * torch.log(sigma).to(self.device)

    def kl_div(self, qz, mask=None, norm=True):
        pz_mean = pz_logvar = torch.zeros(qz.mean.size()).to(self.device)
        kl = utils.normal_kl(qz.mean, qz.logvar, pz_mean, pz_logvar).sum(-1).sum(-1)
        if norm:
            return kl / mask.sum(-1).sum(-1)
        return kl

    def compute_loglik(self, target_y, px, norm=True):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        log_p = utils.log_normal_pdf(
            target, px.mean, px.logvar, mask).sum(-1).sum(-1)
        if norm:
            return log_p / mask.sum(-1).sum(-1)
        return log_p

    def compute_mog_loglik(self, target_y, px):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        loglik = utils.mog_log_pdf(target, px.mean, px.logvar, mask)
        # pdb.set_trace()
        return loglik.sum() / mask.sum()

    def compute_mse(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        return utils.mean_squared_error(target, pred, mask)

    def compute_mae(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        return utils.mean_absolute_error(target, pred, mask)

    def compute_mean_mse(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        return utils.mean_squared_error(target, pred.mean(0), mask)

    def compute_mean_mae(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        return utils.mean_absolute_error(target, pred.mean(0), mask)

    def compute_crps(self, target_y, pred):
        target, mask = target_y[:,:,:self.dim2], target_y[:,:,self.dim2:]
        tgt = torch.masked_select(target, mask.bool())
        mean = torch.masked_select(pred.mean[0], mask.bool())
        std = torch.exp(torch.masked_select(pred.logvar[0], mask.bool()))
        crps_score = np.mean(ps.crps_gaussian(tgt.cpu().detach().numpy(), mu=mean.cpu().detach().numpy(), sig=std.cpu().detach().numpy()))
        return crps_score


    def compute_unsupervised_loss(
        self, context_x, context_y, target_x, target_y, num_samples=1, beta=1.
    ):
        px = Gaussian()

        loss_info = LossInfo()

        # pdb.set_trace()
        mean_res = []
        std_res = []
        mask = []
        value = []
        # pdb.set_trace()
        for chan in range(self.dim):
            X = context_x[:,:]
            y = context_y[:,:,chan]
            c_m = context_y[:,:,chan+self.dim]
            # pdb.set_trace()
            asd = torch.where(c_m.to(torch.bool))[1]
            asd = torch.cat([torch.tensor([0]).to(self.device), asd], 0)
            obslen = torch.max(asd+1).to(torch.int64)
            X = X[:,:obslen]
            y = y[:,:obslen]
            c_m = c_m[:,:obslen]
            X_str = target_x[:,:]
            y_str = target_y[:,:,chan]
            t_m = target_y[:,:,chan+self.dim]
            # pdb.set_trace()
            asd = torch.where(t_m.to(torch.bool))[1]
            asd = torch.cat([torch.tensor([0]).to(self.device), asd], 0)
            obslen = torch.max(asd+1).to(torch.int64)
            # obslen = max(1, obslen)
            X_str = X_str[:,:obslen]
            y_str = y_str[:,:obslen]
            t_m = t_m[:,:obslen]
            mean, std = self.gpr[chan](X[:,:,None],y,c_m,X_str[:,:,None],t_m)
            # pdb.set_trace()
            # mean *= t_m[:,:,None]
            # std *= t_m[:,:,None]

            mean_res.append(mean)
            std_res.append(torch.log(std + 1e-2))
            mask.append(t_m)
            value.append(y_str)
        # pdb.set_trace()
        px.mean = torch.cat(mean_res, 1)
        px.logvar = torch.cat(std_res, 1)
        # pdb.set_trace()
        # pdb.set_trace()
        if px.mean.isnan().any():
            pdb.set_trace()

        mask = torch.cat(mask, 1)
        value = torch.cat(value, 1)
        result_mask = torch.cat([value[:,:,None], mask[:,:,None]], -1)
        
        self.dim2 = 1

        loglik = self.compute_loglik(result_mask, px, self.norm)
        # pdb.set_trace()
        loss_info.loglik = loglik.mean()
        # pdb.set_trace()
        loss_info.mse = self.compute_mse(result_mask, px.mean)

        loss_info.mae = self.compute_mae(result_mask, px.mean)
        loss_info.mean_mse = self.compute_mean_mse(result_mask, px.mean)
        loss_info.mean_mae = self.compute_mean_mae(result_mask, px.mean)
        loss_info.mogloglik = self.compute_mog_loglik(result_mask, px)
        # loss_info.crps = self.compute_crps(result_mask, px)
        loss_info.crps = 0
        loss_info.composite_loss = -loss_info.loglik
        return loss_info
