#__pacage__ = None
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

import sys

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
        self.Wvn_int = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.Wvn_nh = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.bv = nn.Parameter(torch.zeros(self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))


    def forward(self, x):

        vertices, nh_indices, int_indices, nh_edges, int_edges, is_int = x

        # generate vertex signals
        Zc = torch.matmul(vertices, self.Wvc)  # (n_verts, filters)

        # create neighbor signals
        v_Wvn_int = torch.matmul(vertices, self.Wvn_int)  # (n_verts, filters)
        v_Wvn_nh = torch.matmul(vertices, self.Wvn_nh)  # (n_verts, filters)

        if(len(int_edges.size()) != 3):
            int_edges = torch.unsqueeze(int_edges, 2)
            nh_edges = torch.unsqueeze(nh_edges, 2)

        int_part = torch.unsqueeze(int_indices != -1, 2).type(torch.double)
        nh_part = torch.unsqueeze(nh_indices != -1, 2).type(torch.double)

        int_norm = torch.sum(int_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)
        nh_norm = torch.sum(nh_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)

        int_norm[int_norm == 0] = 1
        nh_norm[nh_norm == 0] = 1

        Zn_inter = torch.sum(v_Wvn_int[int_indices] * int_part * int_edges, 1) / int_norm
        Zn_nh = torch.sum(v_Wvn_nh[nh_indices] * nh_part * nh_edges, 1) / nh_norm

        sig = Zc + Zn_inter + Zn_nh

        if self.bias:
            sig += self.bv

        z = F.relu(sig)

        if self.dropout:
            z = F.dropout(z, self.dropout)

        return [z, nh_indices, int_indices, nh_edges, int_edges, is_int]


class DGCN(nn.Module):

    def __init__(self, v_feats, filters, dropout=0.1, bias=True, trainable=True, **kwargs):

        super(DGCN, self).__init__()

        self.v_feats = v_feats
        self.filters = filters
        self.dropout= dropout
        self.bias = bias
        self.trainable = trainable

        self.device = torch.device("cuda")

        self.Wvc_int = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.Wvc_nh = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.Wvn_int = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.Wvn_nh = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.bv_int = nn.Parameter(torch.zeros(self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))
        self.bv_nh = nn.Parameter(torch.zeros(self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))


    def forward(self, x):

        vertices_int, vertices_nh, nh_indices, int_indices, nh_edges, int_edges, is_int = x

        int_atom_pos = (is_int == 1)
        nh_atom_pos = (is_int == 0)

        vertices_int = vertices_int * int_atom_pos
        vertices_nh = vertices_nh * nh_atom_pos

        # generate vertex signals
        Zc_int = torch.matmul(vertices_int, self.Wvc_int)  # (n_verts, filters)
        Zc_nh = torch.matmul(vertices_nh, self.Wvc_nh)  # (n_verts, filters)

        # create neighbor signals
        v_Wvn_int = torch.matmul(vertices_int, self.Wvn_int)  # (n_verts, filters)
        v_Wvn_nh = torch.matmul(vertices_nh, self.Wvn_nh)  # (n_verts, filters)

        if(len(int_edges.size()) != 3):
            int_edges = torch.unsqueeze(int_edges, 2)
            nh_edges = torch.unsqueeze(nh_edges, 2)

        int_part = torch.unsqueeze(int_indices != -1, 2).type(torch.double)
        nh_part = torch.unsqueeze(nh_indices != -1, 2).type(torch.double)

        int_norm = torch.sum(int_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)
        nh_norm = torch.sum(nh_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)

        int_norm[int_norm == 0] = 1
        nh_norm[nh_norm == 0] = 1

        Zn_int = torch.sum(v_Wvn_int[int_indices] * int_part * int_edges, 1) / int_norm
        Zn_nh = torch.sum(v_Wvn_nh[nh_indices] * nh_part * nh_edges, 1) / nh_norm

        sig_int = Zc_int + Zn_int
        sig_nh = Zc_nh + Zn_nh

        if self.bias:
            sig_int += self.bv_int
            sig_nh += self.bv_nh

        z_int = F.relu(sig_int)
        z_nh = F.relu(sig_nh)

        if self.dropout:
            z_int = F.dropout(z_int, self.dropout)
            z_nh = F.dropout(z_nh, self.dropout)

        return [z_int, z_nh, nh_indices, int_indices, nh_edges, int_edges, is_int]



class EGCN(nn.Module):

    def __init__(self, v_feats, filters, dropout=0.1, bias=True, trainable=True, **kwargs):

        super(EGCN, self).__init__()

        self.v_feats = v_feats
        self.filters = filters
        self.dropout= dropout
        self.bias = bias
        self.trainable = trainable

        self.device = torch.device("cuda")

        self.W = nn.Parameter(init.kaiming_uniform_(torch.randn(self.v_feats, self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.bv = nn.Parameter(torch.zeros(self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))


    def forward(self, x):

        vertices_int, vertices_nh, nh_indices, int_indices, nh_edges, int_edges, is_int = x

        int_atom_pos = (is_int == 1)
        nh_atom_pos = (is_int == 0)

        vertices_int = vertices_int * int_atom_pos
        vertices_nh = vertices_nh * nh_atom_pos

        h_ints = []
        h_nhs = []

        for p_feat in range(self.v_feats):

            Z_int = torch.matmul(vertices_int, self.W[p_feat])
            Z_nh = torch.matmul(vertices_nh, self.W[p_feat])

            if(len(int_edges.size()) != 3):
                int_edges = torch.unsqueeze(int_edges, 2)
                nh_edges = torch.unsqueeze(nh_edges, 2)

            int_part = torch.unsqueeze(int_indices != -1, 2).type(torch.double)
            nh_part = torch.unsqueeze(nh_indices != -1, 2).type(torch.double)

            Zn_nh = Z_int[nh_indices] * nh_part * nh_edges
            Zn_int = Z_nh[int_indices] * int_part * int_edges

            Z_int = Z_int.unsqueeze(1)
            Z_nh = Z_nh.unsqueeze(1)

            Z_p_feat_int = torch.sum(torch.sum(Z_int * Zn_nh, dim=1), dim=1).unsqueeze(1)
            Z_p_feat_nh = torch.sum(torch.sum(Z_nh * Zn_int, dim=1), dim=1).unsqueeze(1)

            h_ints.append(Z_p_feat_int)
            h_nhs.append(Z_p_feat_nh)

        h_int = torch.cat(h_ints, dim=1)
        h_nh = torch.cat(h_nhs, dim=1)

        h_int = F.leaky_relu(h_int)
        h_nh = F.leaky_relu(h_nh)

        return [h_int, h_nh, nh_indices, int_indices, nh_edges, int_edges, is_int]


class bGCN(nn.Module):

    def __init__(self, v_feats, filters, dropout=0.1, bias=True, trainable=True, **kwargs):

        super(bGCN, self).__init__()

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

    def __init__(self, v_feats, filters, dropout=0.1, multi_head=3, bias=True, trainable=True, **kwargs):

        super(GAT, self).__init__()

        self.v_feats = v_feats
        self.filters = filters
        self.dropout= dropout
        self.bias = bias
        self.trainable = trainable
        self.multi_head = multi_head

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.Wvc = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))
        self.bv = nn.Parameter(torch.zeros(self.multi_head, self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))
        self.Wvn_int = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))
        self.Wvn_nh = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))
        self.a = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, 2*self.filters, 1, requires_grad=self.trainable)).type(torch.double).to(self.device))


    def forward(self, x):

        vertices, nh_indices, int_indices, nh_edges, int_edges, is_int = x

        head_hidden_node_reprs = []

        for head in range(self.multi_head):
            # generate vertex signals
            Zc = torch.matmul(vertices, self.Wvc[head])  # (n_verts, filters)

            # create neighbor signals
            v_Wvn_int = torch.matmul(vertices, self.Wvn_int[head])  # (n_verts, filters)
            v_Wvn_nh = torch.matmul(vertices, self.Wvn_nh[head])

            if(len(int_edges.size()) != 3):
                int_edges = torch.unsqueeze(int_edges, 2)
                nh_edges = torch.unsqueeze(nh_edges, 2)
            int_part = torch.unsqueeze(int_indices != -1, 2).type(torch.double)
            nh_part = torch.unsqueeze(nh_indices != -1, 2).type(torch.double)

            e_inter = (torch.matmul(torch.cat((v_Wvn_int[int_indices], Zc.unsqueeze(1).repeat(1, 10, 1)), 2), self.a[head]) * int_part * int_edges).squeeze(2)
            alpha_inter = F.softmax(e_inter, 1).unsqueeze(2)

            e_chain = (torch.matmul(torch.cat((v_Wvn_nh[nh_indices], Zc.unsqueeze(1).repeat(1, 10, 1)), 2), self.a[head]) * nh_part * nh_edges).squeeze(2)
            alpha_chain = F.softmax(e_chain, 1).unsqueeze(2)

            int_norm = torch.sum(int_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)
            nh_norm = torch.sum(nh_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)

            int_norm[int_norm == 0] = 1
            nh_norm[nh_norm == 0] = 1

            Zn_inter = torch.sum(v_Wvn_int[int_indices] * int_part * alpha_inter, 1) / int_norm
            Zn_chain = torch.sum(v_Wvn_nh[nh_indices] * nh_part * alpha_chain, 1) / nh_norm

            sig = Zc + Zn_inter + Zn_chain

            if self.bias:
                sig += self.bv[head]

            z = F.relu(sig)

            head_hidden_node_reprs.append(z)

        head_hidden_node_reprs_cat = torch.cat(head_hidden_node_reprs, dim=1)

        # if self.dropout:
        #     z = F.dropout(z, self.dropout)

        return [head_hidden_node_reprs_cat, nh_indices, int_indices, nh_edges, int_edges, is_int]



class DGAT(nn.Module):

    def __init__(self, v_feats, filters, dropout=0.1, multi_head=3, bias=True, trainable=True, **kwargs):

        super(DGAT, self).__init__()

        self.v_feats = v_feats
        self.filters = filters
        self.dropout= dropout
        self.bias = bias
        self.trainable = trainable
        self.multi_head = multi_head

        self.device = torch.device("cuda")

        self.Wvc_int = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device)) # (v_dims, filters)
        self.Wvc_nh = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device)) # (v_dims, filters)
        self.bv_int = nn.Parameter(torch.zeros(self.multi_head, self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))
        self.bv_nh = nn.Parameter(torch.zeros(self.multi_head, self.filters, requires_grad=self.trainable).type(torch.double).to(self.device))
        self.Wvn_int = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.Wvn_nh = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, self.v_feats, self.filters, requires_grad=self.trainable)).type(torch.double).to(self.device))  # (v_dims, filters)
        self.a_int = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, 2*self.filters, 1, requires_grad=self.trainable)).type(torch.double).to(self.device))
        self.a_nh = nn.Parameter(init.kaiming_uniform_(torch.randn(self.multi_head, 2*self.filters, 1, requires_grad=self.trainable)).type(torch.double).to(self.device))


    def forward(self, x):

        vertices_int, vertices_nh, nh_indices, int_indices, nh_edges, int_edges, is_int = x

        int_atom_pos = (is_int == 1)
        nh_atom_pos = (is_int == 0)

        vertices_int = vertices_int * int_atom_pos
        vertices_nh = vertices_nh * nh_atom_pos

        head_hidden_node_reprs_int = []
        head_hidden_node_reprs_nh = []

        for head in range(self.multi_head):
            # generate vertex signals
            Zc_int = torch.matmul(vertices_int, self.Wvc_int[head])  # (n_verts, filters)
            Zc_nh = torch.matmul(vertices_nh, self.Wvc_nh[head])  # (n_verts, filters)

            # create neighbor signals
            v_Wvn_int = torch.matmul(vertices_int, self.Wvn_int[head])  # (n_verts, filters)
            v_Wvn_nh = torch.matmul(vertices_nh, self.Wvn_nh[head])

            if(len(int_edges.size()) != 3):
                int_edges = torch.unsqueeze(int_edges, 2)
                nh_edges = torch.unsqueeze(nh_edges, 2)
            int_part = torch.unsqueeze(int_indices != -1, 2).type(torch.double)
            nh_part = torch.unsqueeze(nh_indices != -1, 2).type(torch.double)

            e_inter = (torch.matmul(torch.cat((v_Wvn_int[int_indices], Zc_int.unsqueeze(1).repeat(1, 10, 1)), 2), self.a_int[head]) * int_part * int_edges).squeeze(2)
            alpha_inter = F.softmax(e_inter, 1).unsqueeze(2)

            e_chain = (torch.matmul(torch.cat((v_Wvn_nh[nh_indices], Zc_nh.unsqueeze(1).repeat(1, 10, 1)), 2), self.a_nh[head]) * nh_part * nh_edges).squeeze(2)
            alpha_chain = F.softmax(e_chain, 1).unsqueeze(2)

            int_norm = torch.sum(int_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)
            nh_norm = torch.sum(nh_indices > -1, 1).unsqueeze(1).to(self.device).type(torch.double)

            int_norm[int_norm == 0] = 1
            nh_norm[nh_norm == 0] = 1

            Zn_int = torch.sum(v_Wvn_int[int_indices] * int_part * alpha_inter, 1) / int_norm
            Zn_nh = torch.sum(v_Wvn_nh[nh_indices] * nh_part * alpha_chain, 1) / nh_norm

            sig_int = Zc_int + Zn_int
            sig_nh = Zc_nh + Zn_nh

            if self.bias:
                sig_int += self.bv_int[head]
                sig_nh += self.bv_nh[head]

            z_int = F.relu(sig_int)
            z_nh = F.relu(sig_nh)

            if self.dropout:
                z_int = F.dropout(z_int, self.dropout)
                z_nh = F.dropout(z_nh, self.dropout)



            head_hidden_node_reprs_int.append(z_int)
            head_hidden_node_reprs_nh.append(z_nh)

        head_hidden_node_reprs_int_cat = torch.cat(head_hidden_node_reprs_int, dim=1)
        head_hidden_node_reprs_nh_cat = torch.cat(head_hidden_node_reprs_nh, dim=1)

        return [head_hidden_node_reprs_int_cat, head_hidden_node_reprs_nh_cat, nh_indices, int_indices, nh_edges, int_edges, is_int]



class SelfAttention(nn.Module):
    def __init__(self, attention_size, heads=3, dropout=0.1, batch_first=False, nonlin="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.dropout = dropout
        self.heads = heads
        self.device = torch.device("cuda")

        self.attention_weights = nn.Parameter(torch.DoubleTensor(self.heads, attention_size).to(self.device))
        self.softmax = nn.Softmax(dim=-1)
        self.nonlin = nonlin

        if self.nonlin == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        init.uniform_(self.attention_weights.data, -0.005, 0.005)

    def forward(self, inputs):

        hidden_reprs = []
        head_scores = []
        for head in range(self.heads):
            ##################################################################
            # STEP 1 - perform dot product
            # of the attention vector and each hidden state
            ##################################################################

            # inputs is a 3D Tensor: batch, len, hidden_size
            # scores is a 2D Tensor: batch, len

            scores = self.non_linearity(inputs.matmul(self.attention_weights[head]))
            scores = self.softmax(scores)

            ##################################################################
            # Step 2 - Weighted sum of hidden states, by the attention scores
            ##################################################################

            # multiply each hidden state with the attention weights
            weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

            # sum the hidden states
            representations = F.normalize(torch.sum(weighted, 0).view(1, -1))

            if self.dropout and self.nonlin != "linear" and self.nonlin != "sigmoid":
                representations = F.dropout(representations, self.dropout)

            hidden_reprs.append(representations)
            head_scores.append(scores)

        hidden_reprs_cat = torch.cat(hidden_reprs, dim=1)

        return hidden_reprs_cat, head_scores


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
