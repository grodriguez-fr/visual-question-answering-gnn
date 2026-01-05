#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: gnn.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-03-05
# 
# This file is part of MGN
# Distributed under terms of the MIT License

import inspect
from collections import OrderedDict

import torch
from torch.nn import Linear as Lin, BatchNorm1d as BN
from torch.nn import Sequential as Seq, ReLU
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import GINConv, GATConv, global_mean_pool

__all__ = ['GIN', 'MLP', 'GAT']

class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(in_channels, out_channels, 2, batch_norm)
            self.convs.append(GINConv(mlp, train_eps=True))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        """"""
        xs = [x]

        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, '
                'lin={})').format(self.__class__.__name__, self.in_channels,
                                  self.out_channels, self.num_layers,
                                  self.batch_norm, self.cat, self.lin)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 dropout=0.0):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.lins = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Lin(in_channels, out_channels))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        for lin, batch_norm in zip(self.lins, self.batch_norms):
            lin.reset_parameters()
            batch_norm.reset_parameters()

    def forward(self, x, *args):
        for i, (lin, bn) in enumerate(zip(self.lins, self.batch_norms)):
            if i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = bn(x) if self.batch_norm else x
        return x

    def __repr__(self):
        return '{}({}, {}, num_layers={}, batch_norm={}, dropout={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_layers, self.batch_norm, self.dropout)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, heads=4, dropout=0.1):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.out_channels = out_channels
        
        self.hidden_per_head = out_channels // heads
        
        self.convs = torch.nn.ModuleList()
        
        self.convs.append(GATConv(in_channels, self.hidden_per_head, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(out_channels, self.hidden_per_head, heads=heads, dropout=dropout))

    def forward(self, x, edge_index, edge_attr=None, batch=None, return_attention=False):
        attention_weights_list = []
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if return_attention:
                x, (edge_index_attn, alpha) = conv(x, edge_index, return_attention_weights=True)
                attention_weights_list.append((edge_index_attn, alpha))
            else:
                x = conv(x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        if return_attention:
            return x, attention_weights_list
        return x