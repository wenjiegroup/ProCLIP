import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn import init
from collections.abc import Sequence

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input,train=False):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden


class PPIClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(PPIClassifier, self).__init__()
        self.predict_module  =   MultiLayerPerceptron(2560, hidden_dims, batch_norm=batch_norm, dropout=0.3)
        self.predict_module_c  =   MultiLayerPerceptron(3072, [1024], batch_norm=batch_norm, dropout=0.0)
        self.decoder  =   MultiLayerPerceptron(hidden_dims[-1], [512, 2], batch_norm=batch_norm)

    def forward(self, pro_emb, train=False):
        pro_emb_0 = pro_emb[:,:2560]
        pro_emb_1 = pro_emb[:,2560:5120]
        pro_emb_complex = pro_emb[:,5120:]

        pro_emb_0 = self.predict_module(pro_emb_0)
        pro_emb_1 = self.predict_module(pro_emb_1)
        complex_rep = self.predict_module_c(pro_emb_complex) * pro_emb_0 * pro_emb_1
        output = self.decoder(complex_rep)
        return output
