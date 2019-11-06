#__pacage__ = None
import logging
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import sys
import time

class GCN(nn.Module):

    def __init__(self, v_feats, filters, dropout=0.1, bias=True, trainable=True, **kwargs):

        super(GCN, self).__init__()

        self.v_feats = v_feats
        self.filters = filters
        self.dropout= dropout
        self.bias = bias
        self.trainable = trainable

        self.device = torch.device("cuda")

        self.Wvc = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.Wvn = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.bv = nn.Parameter(torch.zeros(self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))


    def forward(self, x):

        vertices, nh_indices, int_indices, nh_edges, int_edges = x

        # generate vertex signals
        Zc = torch.matmul(vertices, self.Wvc)  # (n_verts, filters)

        # create neighbor signals
        v_Wvn = torch.matmul(vertices, self.Wvn)  # (n_verts, filters)

        if(len(int_edges.size()) != 3):
            int_edges = torch.unsqueeze(int_edges, 2)
            nh_edges = torch.unsqueeze(nh_edges, 2)
        int_part = torch.unsqueeze(int_indices != -1, 2).type(torch.double)
        nh_part = torch.unsqueeze(nh_indices != -1, 2).type(torch.double)

        Zn_inter = torch.sum(v_Wvn[int_indices] * int_part * int_edges, 1) / torch.sum(int_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)
        Zn_chain = torch.sum(v_Wvn[nh_indices] * nh_part * nh_edges, 1) / torch.sum(nh_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)

        sig = Zc + Zn_inter + Zn_chain

        if self.bias:
            sig += self.bv

        z = F.relu(sig)

        if self.dropout:
            z = F.dropout(z, self.dropout)

        return [z, nh_indices, int_indices, nh_edges, int_edges]



class bGCN(nn.Module):

    def __init__(self, v_feats, filters, dropout=0.1, bias=True, trainable=True, **kwargs):

        super(GCN, self).__init__()

        self.v_feats = v_feats
        self.filters = filters
        self.dropout= dropout
        self.bias = bias
        self.trainable = trainable

        self.device = torch.device("cuda")

        self.Wvc = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.Wvn = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.bv = nn.Parameter(torch.zeros(self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))

    def forward(self, x):

        vertices, nh_indices, int_indices, nh_edges, int_edges = x

        indices = torch.cat((nh_indices, int_indices), 1)
        edges = torch.cat((nh_edges, int_edges), 1)

        # generate vertex signals
        Zc = torch.matmul(vertices, self.Wvc)  # (n_verts, filters)

        # create neighbor signals
        v_Wvn = torch.matmul(vertices, self.Wvn)  # (n_verts, filters)

        if(len(int_edges.size()) != 3):
            int_edges = torch.unsqueeze(int_edges, 2)
            nh_edges = torch.unsqueeze(nh_edges, 2)
            edges = torch.cat((nh_edges, int_edges), 1)

        int_part = torch.unsqueeze(int_indices != -1, 2).type(torch.double)
        nh_part = torch.unsqueeze(nh_indices != -1, 2).type(torch.double)
        part = torch.cat((nh_part, int_part), 1)

        Zn = torch.sum(v_Wvn[indices] * part * edges, 1) / torch.sum(indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)

        sig = Zc + Zn

        if self.bias:
            sig += self.bv

        z = F.relu(sig)

        if self.dropout:
            z = F.dropout(z, self.dropout)

        return [z, nh_indices, int_indices, nh_edges, int_edges]


class GAT(nn.Module):

    def __init__(self, v_feats, filters, dropout=0.1, bias=True, trainable=True, **kwargs):

        super(GAT, self).__init__()

        self.v_feats = v_feats
        self.filters = filters
        self.dropout= dropout
        self.bias = bias
        self.trainable = trainable

        self.device = torch.device("cuda")

        self.Wvc = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.bv = nn.Parameter(torch.zeros(self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))
        self.Wvn = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.a = nn.Parameter(init.kaiming_uniform_(torch.randn(2*self.filters, 1, requires_grad=self.trainable)).type(torch.double).to(self.device))


    def forward(self, x):

        vertices, nh_indices, int_indices, nh_edges, int_edges = x

        # generate vertex signals
        Zc = torch.matmul(vertices, self.Wvc)  # (n_verts, filters)

        # create neighbor signals
        v_Wvn = torch.matmul(vertices, self.Wvn)  # (n_verts, filters)

        if(len(int_edges.size()) != 3):
            int_edges = torch.unsqueeze(int_edges, 2)
            nh_edges = torch.unsqueeze(nh_edges, 2)
        int_part = torch.unsqueeze(int_indices != -1, 2).type(torch.double)
        nh_part = torch.unsqueeze(nh_indices != -1, 2).type(torch.double)

        e_inter = (torch.matmul(torch.cat((v_Wvn[int_indices], Zc.unsqueeze(1).repeat(1, 10, 1)), 2), self.a) * int_part * int_edges).squeeze(2)
        alpha_inter = F.softmax(e_inter, 1).unsqueeze(2)

        e_chain = (torch.matmul(torch.cat((v_Wvn[nh_indices], Zc.unsqueeze(1).repeat(1, 10, 1)), 2), self.a) * nh_part * nh_edges).squeeze(2)
        alpha_chain = F.softmax(e_chain, 1).unsqueeze(2)

        Zn_inter = torch.sum(v_Wvn[int_indices] * int_part * alpha_inter, 1) / torch.sum(int_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)
        Zn_chain = torch.sum(v_Wvn[nh_indices] * nh_part * alpha_chain, 1) / torch.sum(nh_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)

        sig = Zc + Zn_inter + Zn_chain

        if self.bias:
            sig += self.bv

        z = F.relu(sig)

        if self.dropout:
            z = F.dropout(z, self.dropout)

        return [z, nh_indices, int_indices, nh_edges, int_edges]



class Dense(nn.Module):
    def __init__(self, in_dims, out_dims, nonlin="relu", merge=False, dropout=0.2, trainable=True, **kwargs):

        super(Dense, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.nonlin = nonlin
        self.dropout = dropout
        self.trainable = trainable
        self.merge = merge
        self.device = torch.device("cuda")

        self.W = nn.Parameter(init.kaiming_uniform_(torch.randn(in_dims, out_dims, requires_grad=self.trainable)).type(torch.double).to(self.device))
        self.b = nn.Parameter(torch.zeros(out_dims, requires_grad=self.trainable).type(torch.double).to(self.device))


    def forward(self, x):
        out_dims = self.in_dims if self.out_dims is None else self.out_dims

        Z = torch.matmul(x, self.W) + self.b
        if self.nonlin == "relu":
            Z = F.relu(Z)

        if self.nonlin == "sigmoid":
            Z = torch.sigmoid(Z)

        if self.merge:
            Z = torch.sum(Z, 0)

        if self.dropout and self.nonlin != "linear" and self.nonlin != "sigmoid":
            Z = F.dropout(Z, self.dropout)

        return Z
