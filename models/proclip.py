import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
import math
import scipy.sparse as spp
from .esm2net import ESM2_encoder
from bidirectional_cross_attention import *

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.t_layers = 3
        self.encoder_q = ESM2_encoder(args, layer = 34)

        self.projector_in = nn.Sequential(
            nn.Linear(2560, 1536, bias=False),
            nn.LayerNorm(1536),
            nn.Dropout(0.05))

        self.joint_cross_attn = BidirectionalCrossAttentionTransformer(
            dim = 1536,
            depth = self.t_layers,
            context_dim = 1536
        )


    def get_padding_mask(self, seq_len, max_len=1024):
        padding_mask = torch.arange(max_len).view(1, -1).repeat(seq_len.size(0), 1) # B x L
        padding_mask = padding_mask.to(seq_len.device)
        padding_mask = padding_mask >= seq_len.view(-1, 1)
        padding_mask.requres_grad = False
        return padding_mask     # B x L

    def get_pro_rep(self, encs, lens):
        padding_mask = self.get_padding_mask(lens, max_len=encs.size(1))
        rep = encs * (1.-padding_mask.type_as(encs)).unsqueeze(-1)
        rep = torch.sum(rep, dim=1)
        rep = torch.div(rep, lens.unsqueeze(-1))
        return rep

    def forward(self, batch_tokens, masked_tokens, masked_pos, protein_len, result_dict = None, return_type = 'protein_rep', compute_single=True):
        if compute_single:
            result_dict_A = self.encoder_q(
                batch_tokens['chain_A'],
                masked_tokens['chain_A'],
                masked_pos['chain_A'],
                protein_len['chain_A']
                )
            seq_feature_mean = result_dict_A[return_type]['mean_rep'].squeeze(1)
            seq_feature_all = result_dict_A['all_feat']
            return {'mean_rep': seq_feature_mean, 'all_feat' : seq_feature_all}
        else:
            seq_feature_A = result_dict['chain_A_mean']
            seq_feature_B = result_dict['chain_B_mean']
            seq_feature = torch.cat((seq_feature_A, seq_feature_B), dim=-1)
            feat_a, feat_b = self.joint_cross_attn(self.projector_in(result_dict['chain_A_all']), self.projector_in(result_dict['chain_B_all']))
            feat_a = self.get_pro_rep(feat_a, protein_len['chain_A'])
            feat_b = self.get_pro_rep(feat_b, protein_len['chain_B'])
            seq_feat_cross = torch.cat((feat_a, feat_b), -1).squeeze(1)
            return torch.cat((seq_feature, seq_feat_cross),dim=-1)

