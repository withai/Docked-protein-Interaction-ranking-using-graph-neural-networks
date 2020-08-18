import sys

import torch
import torch.optim as optim
import torch.nn.functional as F

from nn import GAT, DGAT, GCN, DGCN, bGCN, EGCN, Dense, SelfAttention
from loss import BatchRankingLoss


# GNN33, MS: 15 19, PS: 13 19.
class GNN35(torch.nn.Module):
    def __init__(self, lr=0.0001, dropout=0.4, weight_decay=0.01):
        super(GNN35, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1
        self.multi_label = False

        self.conv1 = DGAT(v_feats=11, filters=16, dropout=0.3, multi_head=6)
        self.conv2 = DGAT(v_feats=16*6, filters=32, dropout=0.3, multi_head=6)
        self.conv3 = DGAT(v_feats=32*6, filters=64, dropout=0.3, multi_head=6)
        self.dense1 = Dense(in_dims=6*2*64, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001, weight_decay=0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_int, x_nh = x[0], x[1]

        x = torch.cat((x_int, x_nh), dim=1)
        
        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)
