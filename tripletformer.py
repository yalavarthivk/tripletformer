import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import layers
import pdb

class Gaussian:
    mean = None
    logvar = None

class LossInfo:
    loglik = None
    mse = None
    mae = None
    composite_loss = None


class TRIPLETFORMER(nn.Module):

    def __init__(
        self,
        input_dim=41,
        enc_num_heads=4,
        dec_num_heads=4,
        num_ref_points=128,
        mse_weight=1.,
        norm=True,
        imab_dim = 128,
        cab_dim = 128,
        decoder_dim = 128,
        n_layers=2,
        device='cuda'):
        super().__init__()
        self.dim=input_dim
        self.enc_num_heads=enc_num_heads
        self.dec_num_heads=dec_num_heads
        self.num_ref_points=num_ref_points
        self.mse_weight=mse_weight
        self.norm=norm
        self.imab_dim = imab_dim
        self.cab_dim = cab_dim
        self.decoder_dim = decoder_dim
        self.n_layers=n_layers
        self.device=device
        self.enc = layers.Encoder(self.dim, self.imab_dim, self.n_layers, self.num_ref_points, self.enc_num_heads, device=device)
        self.dec_att = layers.Decoder_att(self.dim, self.imab_dim, self.cab_dim, self.dec_num_heads, device=device)
        self.O = layers.output(self.cab_dim, self.decoder_dim, device=device)

    def encode(self, context_x, context_w, target_x):
        mask = context_w[:, :, self.dim:]
        X = context_w[:, :, :self.dim]
        Z_e, mk_e = self.enc(context_x, X, mask)
        return Z_e, mk_e

    def decode(self, Z_e, mk_e, target_context, target_mask):
        px = Gaussian()

        Z_d = self.dec_att(Z_e, mk_e, target_context, target_mask)
        dec_out = self.O(Z_d)
        px.mean = dec_out[:,:,0:1]
        px.logvar = torch.log(1e-8 + F.softplus(dec_out[:, :, 1:]))
        return px

    def get_interpolation(self, context_x, context_y, target_x, target_context, target_mask):
        Z_e, mk_e = self.encode(context_x, context_y, target_x)
        px = self.decode(Z_e, mk_e, target_context, target_mask)
        return px

    def compute_loglik(self, target_y, px, norm=True):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        log_p = utils.log_normal_pdf(
            target, px.mean, px.logvar, mask).sum(-1).sum(-1)
        if norm:
            return log_p / mask.sum(-1).sum(-1)
        return log_p

    def compute_mse(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        return utils.mean_squared_error(target, pred, mask)

    def compute_mae(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        return utils.mean_absolute_error(target, pred, mask)


    def compute_unsupervised_loss(
        self, context_x, context_y, target_x, target_y, num_samples=1, beta=1.
    ):
        loss_info = LossInfo()

        tau = target_x[:,:,None].repeat(1,1,self.dim)
        U = target_y[:,:,:self.dim]
        mk = target_y[:,:,self.dim:]
        C = torch.ones(mk.size(), dtype=torch.int64).cumsum(-1) - 1
        C = C.to(self.device)

        mk_bool = mk.to(torch.bool)

        full_len = tau.size(1)*self.dim
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0)

        tau = torch.stack([pad(r[m]) for r, m in zip(tau, mk_bool)]).contiguous()
        U = torch.stack([pad(r[m]) for r, m in zip(U, mk_bool)]).contiguous()
        mk = torch.stack([pad(r[m]) for r, m in zip(mk, mk_bool)]).contiguous()
        C = torch.stack([pad(r[m]) for r, m in zip(C, mk_bool)]).contiguous()
        C_ = C
        C = torch.nn.functional.one_hot(C, num_classes =self.dim)

        target_context = torch.cat([tau[:,:,None], C], -1).contiguous()
        target_mask = torch.stack([C_, mk], -1)

        obs_len = torch.max(target_mask[:,:,1].sum(-1)).to(torch.int64)
        target_context = target_context[:, :obs_len]
        target_mask = target_mask[:, :obs_len]
        target_vals = U[:,:obs_len]

        mask = torch.cat([target_vals[:,:,None], target_mask[:,:,1:]],-1)

        px = self.get_interpolation(context_x, context_y, target_x, target_context, target_mask)
        
        self.dim2 = 1

        loglik = self.compute_loglik(mask, px, self.norm)
        loss_info.loglik = loglik.mean()
        loss_info.mse = self.compute_mse(mask, px.mean)
        loss_info.mae = self.compute_mae(mask, px.mean)
        loss_info.composite_loss = -loss_info.loglik + self.mse_weight * loss_info.mse
        return loss_info


