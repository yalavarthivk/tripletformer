import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention, IMAB
import pdb

class output(nn.Module):
    def __init__(self, dim=128, nkernel=128, nlayers=1, device='cuda'):
        super(output, self).__init__()
        self.device=device
        self.nlayers=nlayers
        self.dense = nn.ModuleList()
        self.dim = dim
        self.nkernel=nkernel
        for i in range(self.nlayers):
            self.dense.append(nn.Linear(dim, nkernel))
            dim = nkernel
        self.dense.append(nn.Linear(dim,2))
        self.relu = nn.ReLU()
    def forward(self, Z_d):
        for i in range(self.nlayers):
            hid = self.relu(self.dense[i](Z_d))
        parameters = self.dense[-1](hid)
        return parameters



class Encoder(nn.Module):
    def __init__(self, dim = 41, nkernel = 128, n_layers=3, n_ref_points=128, enc_num_heads = 4, device="cuda"):
        super(Encoder, self).__init__()
        self.dim = dim+2
        self.nheads = enc_num_heads
        self.iFF = nn.Linear(self.dim, nkernel)
        self.n_layers = n_layers
        self.SAB = nn.ModuleList()
        for i in range(self.n_layers):
            self.SAB.append(IMAB(nkernel, nkernel, self.nheads, n_ref_points))
        
        self.relu = nn.ReLU()

    def forward(self, context_x, value, mask):

        seq_len = context_x.size(-1)
        ndims = value.shape[-1]

        T = context_x[:,:,None].repeat(1,1,mask.shape[-1])
        C = torch.cumsum(torch.ones_like(value).to(torch.int64), -1) - 1
        mk_bool = mask.to(torch.bool)
        full_len = seq_len*mask.size(-1)
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0)

        T_ = torch.stack([pad(r[m]) for r, m in zip(T, mk_bool)]).contiguous()
        U_ = torch.stack([pad(r[m]) for r, m in zip(value, mk_bool)]).contiguous()
        C_ = torch.stack([pad(r[m]) for r, m in zip(C, mk_bool)]).contiguous()
        mk_ = torch.stack([pad(r[m]) for r, m in zip(mask, mk_bool)]).contiguous()
        
        obs_len = torch.max(mk_.sum(-1)).to(torch.int64)
        T_ = T_[:, :obs_len]
        U_ = U_[:, :obs_len]
        C_ = C_[:,:obs_len]
        C_ = torch.nn.functional.one_hot(C_.to(torch.int64), num_classes=self.dim-2)
        mk_ = mk_[:,:obs_len]

        X = torch.cat([C_, T_.unsqueeze(-1), U_.unsqueeze(-1)], -1).contiguous()
        X = X*mk_[:,:,None].repeat(1,1,X.size(-1)).contiguous()
        mk_ = mk_.unsqueeze(-1).contiguous()

        # iFF layer
        Y_e = self.relu(self.iFF(X))
        Y_e = Y_e*mk_.repeat(1,1, Y_e.shape[-1])


        attn_mask = mk_[:,:,0]
        attn_input = Y_e
        for i in range(self.n_layers):
            Z_e = self.SAB[i](attn_input, attn_input, mask1=attn_mask, mask2=attn_mask)[0]
            Z_e = Z_e*mk_.repeat([1, 1, Z_e.shape[-1]])
            Z_e = Z_e + attn_input
            attn_input = Z_e

        return Z_e, mk_



class Decoder_att(nn.Module):
    def __init__(self, query_dim=None, key_dim=128, nkernel=128,  dec_num_heads = 2, device="cuda"):
        super(Decoder_att, self).__init__()
        self.dim = query_dim+1
        self.key_dim = key_dim
        self.device=device
        self.oFF = nn.Linear(self.dim, nkernel)
        self.input_dense_key = nn.Linear(self.key_dim,nkernel)
        self.nheads = dec_num_heads
        self.CAB = MultiHeadAttention(nkernel, self.nheads)
        self.relu = nn.ReLU()
        self.res_con = nn.Linear(nkernel, nkernel)

    def forward(self, Z_e, mk_e, W, target_mask):

        mk_d = target_mask[:,:,1:]
        attn_mask = mk_d.matmul(mk_e.transpose(-2,-1).contiguous())
        Y_d = self.relu(self.oFF(W))
        Z_e_ = self.relu(self.input_dense_key(Z_e))

        q_len = Y_d.size(1)

        attn_ = []

        for i in range(q_len//1000+1):
            q = Y_d[:,i*1000:(i+1)*1000,:]
            attn_mask_i = attn_mask[:,i*1000:(i+1)*1000,:]
            attn_.append(self.CAB(q,Z_e_,Z_e_, mask = attn_mask_i)[0])
        Z_d = torch.cat(attn_, 1)

        Z_d = Z_d*mk_d.repeat(1,1,Z_d.shape[-1])
        Z_d_ = self.relu(Z_d + Y_d)

        Z_d__ = self.res_con(Z_d_)
        Z_d__ *= mk_d.repeat(1,1,Z_d__.shape[-1])
        Z_d__ += Z_d_
        Z_d__ = self.relu(Z_d__)

        return Z_d__