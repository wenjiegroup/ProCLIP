import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
import myesm
from myesm import ESM2

class ESM2_encoder(nn.Module):
    def __init__(self, args, layer = 34, dim = 2560, head = 40):
        super().__init__()
        default_alphabet = myesm.Alphabet.default_alphabet()
        self.seqEncoder = ESM2(
            num_layers = layer,
            embed_dim = dim,
            attention_heads = head,
            alphabet = default_alphabet,
            token_dropout = True,
        )

    def cal_bert_loss(self, predict, pos, target):
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        predict = predict * pos.unsqueeze(2).repeat(1, 1, 33)
        ece_loss = ((criterion(predict.transpose(1,2), target) * pos).sum() / pos.sum())
        return ece_loss

    def get_padding_mask(self, seq_len, max_len):

        padding_mask = torch.arange(max_len).view(1, -1).repeat(seq_len.size(0), 1) # B x L
        padding_mask = padding_mask.to(seq_len.device)
        padding_mask = padding_mask >= seq_len.view(-1, 1)
        padding_mask.requres_grad = False
        return padding_mask

    def get_pro_rep(self, encs, lens):

        padding_mask = self.get_padding_mask(lens, max_len=encs.size(1))
        rep = encs * (1.-padding_mask.type_as(encs)).unsqueeze(-1)

        mean_rep = torch.sum(rep, dim=1)
        mean_rep = torch.div(mean_rep, lens.unsqueeze(-1))
        maxrep, _ = torch.max(rep, dim=1)
        minrep, _ = torch.min(rep, dim=1)
        return {'mean_rep': mean_rep, 'max_rep': maxrep, 'min_rep': minrep}

    def logtis_ce_loss(self, x, masked_pos, masked_tokens):
        logits = self.seqEncoder.pre_logits(x)
        seq_loss = self.cal_bert_loss(logits, masked_pos, masked_tokens)
        return seq_loss

    def forward(self, batch_tokens, masked_tokens, masked_pos, protein_len, m_repr_layers = 33):
        #print(batch_tokens.shape, masked_tokens.shape, masked_pos.shape)
        m_repr_layers = 33
        output = self.seqEncoder(tokens = batch_tokens, repr_layers = [m_repr_layers], return_contacts=False)
        all_feat = output["representations"][m_repr_layers]
        protein_rep = self.get_pro_rep(all_feat, protein_len)

        return {'protein_rep': protein_rep, 'all_feat': all_feat}






