import sys

import torch
import torch.optim as optim
import torch.nn.functional as F

from nn import GAT, DGAT, GCN, DGCN, bGCN, EGCN, Dense, SelfAttention
from loss import BatchRankingLoss

class GNN0(torch.nn.Module):
    def __init__(self):
        super(GNN0, self).__init__()
        self.conv1 = GCN(v_feats=11, filters=16, dropout=0.1)
        self.conv2 = GCN(v_feats=16, filters=32, dropout=0.1)
        self.conv3 = GCN(v_feats=32, filters=64, dropout=0.1)
        self.conv4 = GCN(v_feats=64, filters=128, dropout=0.1)
        self.conv5 = GCN(v_feats=128, filters=128, dropout=0.1)
        self.dense1 = Dense(in_dims=128, out_dims=256, dropout=0.1)
        self.dense2 = Dense(in_dims=256, out_dims=128, dropout=0.1)
        self.dense3 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN1(torch.nn.Module):
    def __init__(self):
        super(GNN1, self).__init__()
        self.conv1 = GCN(v_feats=11, filters=16, dropout=0.1)
        self.conv2 = GCN(v_feats=16, filters=32, dropout=0.1)
        self.conv3 = GCN(v_feats=32, filters=64, dropout=0.1)
        self.conv4 = GCN(v_feats=64, filters=128, dropout=0.1)
        self.conv5 = GCN(v_feats=128, filters=128, dropout=0.1)
        self.selfAtt1 = SelfAttention(attention_size=128, batch_first=False, non_linearity="tanh")
        self.dense1 = Dense(in_dims=128, out_dims=256, dropout=0.1)
        self.dense2 = Dense(in_dims=256, out_dims=128, dropout=0.1)
        self.dense3 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN2(torch.nn.Module):
    def __init__(self):
        super(GNN2, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=3)
        self.conv2 = GAT(v_feats=48, filters=64, dropout=0.1, multi_head=3)
        self.dense1 = Dense(in_dims=3*64, out_dims=128, dropout=0.1)
        self.dense2 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN3(torch.nn.Module):
    def __init__(self):
        super(GNN3, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=3)
        self.conv2 = GAT(v_feats=48, filters=64, dropout=0.1, multi_head=3)
        self.selfAtt1 = SelfAttention(attention_size=3*64, heads=3, batch_first=False, nonlin="tanh")
        self.dense1 = Dense(in_dims=3*64*3, out_dims=128, dropout=0.1)
        self.dense2 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN4(torch.nn.Module):
    def __init__(self):
        super(GNN4, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=3)
        self.conv2 = GAT(v_feats=48, filters=64, dropout=0.1, multi_head=3)
        self.selfAtt1 = SelfAttention(attention_size=3*64, heads=3, batch_first=False, nonlin="tanh")
        self.dense1 = Dense(in_dims=3*64*3, out_dims=128, dropout=0.1)
        self.dense2 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        loss = BatchRankingLoss()
        return loss(output, target)


class GNN5(torch.nn.Module):
    def __init__(self):
        super(GNN5, self).__init__()
        self.conv1 = GCN(v_feats=11, filters=16, dropout=0.1)
        self.dense1 = Dense(in_dims=16, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)

class GNN6(torch.nn.Module):
    def __init__(self):
        super(GNN6, self).__init__()
        self.conv1 = GCN(v_feats=11, filters=16, dropout=0.1)
        self.dense1 = Dense(in_dims=16, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)

class GNN7(torch.nn.Module):
    def __init__(self):
        super(GNN7, self).__init__()
        self.conv1 = GCN(v_feats=11, filters=16, dropout=0.1)
        self.selfAtt1 = SelfAttention(attention_size=16, batch_first=False, non_linearity="tanh")
        self.dense1 = Dense(in_dims=16, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)

class GNN8(torch.nn.Module):
    def __init__(self):
        super(GNN8, self).__init__()
        self.conv1 = DGCN(v_feats=11, filters=16, dropout=0.1)
        self.selfAtt1_int = SelfAttention(attention_size=16, batch_first=False, non_linearity="tanh")
        self.selfAtt1_nh = SelfAttention(attention_size=16, batch_first=False, non_linearity="tanh")
        self.dense1 = Dense(in_dims=16*2, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x_int, x_nh = x[0], x[1]

        x_int, _ = self.selfAtt1_int(x_int)
        x_nh, _ = self.selfAtt1_nh(x_nh)

        x = torch.cat((x_int, x_nh), dim=1)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)

class GNN9(torch.nn.Module):
    def __init__(self):
        super(GNN9, self).__init__()
        self.conv1 = GCN(v_feats=11, filters=128, dropout=0.1)
        self.selfAtt1 = SelfAttention(attention_size=128, batch_first=False, non_linearity="tanh")
        self.dense1 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN10(torch.nn.Module):
    def __init__(self):
        super(GNN10, self).__init__()
        self.conv1 = DGCN(v_feats=11, filters=16, dropout=0.1)
        self.conv2 = DGCN(v_feats=16, filters=32, dropout=0.1)
        self.selfAtt1_int = SelfAttention(attention_size=32, batch_first=False, non_linearity="tanh")
        self.selfAtt1_nh = SelfAttention(attention_size=32, batch_first=False, non_linearity="tanh")
        self.dense1 = Dense(in_dims=32*2, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_int, x_nh = x[0], x[1]

        x_int, _ = self.selfAtt1_int(x_int)
        x_nh, _ = self.selfAtt1_nh(x_nh)

        x = torch.cat((x_int, x_nh), dim=1)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN11(torch.nn.Module):
    def __init__(self):
        super(GNN11, self).__init__()
        self.conv1 = EGCN(v_feats=11, filters=32, dropout=0.1)
        self.selfAtt1_int = SelfAttention(attention_size=11, batch_first=False, non_linearity="tanh")
        self.selfAtt1_nh = SelfAttention(attention_size=11, batch_first=False, non_linearity="tanh")
        self.dense1 = Dense(in_dims=11*2, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x_int, x_nh = x[0], x[1]

        # print(x_int.size())
        # print(x_nh.size())

        x_int, _ = self.selfAtt1_int(x_int)
        x_nh, _ = self.selfAtt1_nh(x_nh)

        x = torch.cat((x_int, x_nh), dim=1)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN12(torch.nn.Module):
    def __init__(self):
        super(GNN12, self).__init__()
        self.conv1 = DGCN(v_feats=11, filters=256, dropout=0.2)
        self.selfAtt1_int = SelfAttention(attention_size=256, batch_first=False, nonlin="tanh", dropout=0.2)
        self.selfAtt1_nh = SelfAttention(attention_size=256, batch_first=False, nonlin="tanh", dropout=0.2)
        self.dense1 = Dense(in_dims=256*2, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x_int, x_nh = x[0], x[1]

        x_int, _ = self.selfAtt1_int(x_int)
        x_nh, _ = self.selfAtt1_nh(x_nh)

        x = torch.cat((x_int, x_nh), dim=1)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN13(torch.nn.Module):
    def __init__(self):
        super(GNN13, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = EGCN(v_feats=11, filters=32, dropout=0.1)
        self.conv2 = EGCN(v_feats=11, filters=64, dropout=0.1)
        self.selfAtt1_int = SelfAttention(attention_size=11, heads=3, batch_first=False, nonlin="tanh")
        self.selfAtt1_nh = SelfAttention(attention_size=11, heads=3, batch_first=False, nonlin="tanh")
        self.dense1 = Dense(in_dims=11*2*3, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_int, x_nh = x[0], x[1]

        # print(x_int.size())
        # print(x_nh.size())

        x_int, _ = self.selfAtt1_int(x_int)
        x_nh, _ = self.selfAtt1_nh(x_nh)

        x = torch.cat((x_int, x_nh), dim=1)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN14(torch.nn.Module):
    def __init__(self):
        super(GNN14, self).__init__()
        self.conv1 = DGCN(v_feats=11, filters=16, dropout=0.1)
        self.conv2 = DGCN(v_feats=16, filters=32, dropout=0.1)
        self.selfAtt1 = SelfAttention(attention_size=32*2, batch_first=False, nonlin="tanh")
        self.dense1 = Dense(in_dims=32*2, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_int, x_nh = x[0], x[1]

        x = torch.cat((x_int, x_nh), dim=1)

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)

class GNN15(torch.nn.Module):
    def __init__(self):
        super(GNN15, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = DGAT(v_feats=11, filters=16, head=3, dropout=0.1)
        self.selfAtt1 = SelfAttention(attention_size=16*2*3, heads=3, batch_first=False, nonlin="tanh")
        self.dense1 = Dense(in_dims=16*2*3*3, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x_int, x_nh = x[0], x[1]

        x = torch.cat((x_int, x_nh), dim=1)

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN16(torch.nn.Module):
    def __init__(self):
        super(GNN16, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=6)
        self.conv2 = GAT(v_feats=6*16, filters=64, dropout=0.1, multi_head=6)
        self.dense1 = Dense(in_dims=6*64, out_dims=128, dropout=0.1)
        self.dense2 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN17(torch.nn.Module):
    def __init__(self):
        super(GNN17, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=9)
        self.conv2 = GAT(v_feats=9*16, filters=64, dropout=0.1, multi_head=9)
        self.dense1 = Dense(in_dims=9*64, out_dims=128, dropout=0.1)
        self.dense2 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN18(torch.nn.Module):
    def __init__(self):
        super(GNN18, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=9)
        self.conv2 = GAT(v_feats=9*16, filters=64, dropout=0.1, multi_head=9)
        self.dense1 = Dense(in_dims=9*64, out_dims=128, dropout=0.1)
        self.dense2 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.mse_loss(output, target, reduction=reduction)


class GNN19(torch.nn.Module):
    def __init__(self):
        super(GNN19, self).__init__()
        self.ranking = True
        self.decoys_per_cat = 2

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=3)
        self.conv2 = GAT(v_feats=48, filters=64, dropout=0.1, multi_head=3)
        self.selfAtt1 = SelfAttention(attention_size=3*64, batch_first=False, nonlin="tanh")
        self.dense1 = Dense(in_dims=3*64, out_dims=128, dropout=0.1)
        self.dense2 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        total = 0
        for parameter in self.parameters():
            total += 1
        print(total)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        loss = BatchRankingLoss(decoys_per_complex=6)
        return loss(output, target)


class GNN20(torch.nn.Module):
    def __init__(self):
        super(GNN20, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 2

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=4)
        self.conv2 = GAT(v_feats=16*4, filters=64, dropout=0.1, multi_head=4)
        self.selfAtt1 = SelfAttention(attention_size=4*64, heads=3, batch_first=False, nonlin="tanh")
        self.dense1 = Dense(in_dims=4*64*3, out_dims=128, dropout=0.1)
        self.dense2 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.mse_loss(output, target, reduction=reduction)

## Selected for hyper-parameter tuning.
class GNN21(torch.nn.Module):
    def __init__(self):
        super(GNN21, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.0, multi_head=6)
        self.conv2 = GAT(v_feats=6*16, filters=24, dropout=0.0, multi_head=6)
        self.dense1 = Dense(in_dims=6*24, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN22(torch.nn.Module):
    def __init__(self):
        super(GNN22, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GCN(v_feats=11, filters=16, dropout=0.0)
        self.conv2 = GCN(v_feats=16, filters=32, dropout=0.0)
        self.conv3 = GCN(v_feats=32, filters=64, dropout=0.0)
        self.conv4 = GCN(v_feats=64, filters=128, dropout=0.0)
        self.conv5 = GCN(v_feats=128, filters=128, dropout=0.0)
        self.dense1 = Dense(in_dims=128, out_dims=256, dropout=0.0)
        self.dense2 = Dense(in_dims=256, out_dims=128, dropout=0.0)
        self.dense3 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN23(torch.nn.Module):
    def __init__(self):
        super(GNN23, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 2

        self.conv1 = GCN(v_feats=11, filters=16, dropout=0.1)
        self.conv2 = GCN(v_feats=16, filters=32, dropout=0.1)
        self.conv3 = GCN(v_feats=32, filters=64, dropout=0.1)
        self.conv4 = GCN(v_feats=64, filters=128, dropout=0.1)
        self.conv5 = GCN(v_feats=128, filters=128, dropout=0.1)
        self.selfAtt1 = SelfAttention(attention_size=128, batch_first=False, heads=6, nonlin="tanh")
        self.dense1 = Dense(in_dims=128*6, out_dims=256, dropout=0.1)
        self.dense2 = Dense(in_dims=256, out_dims=128, dropout=0.1)
        self.dense3 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        parameters = 0
        for parameter in self.parameters():
            parameters += 1
        print(parameters)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


## Selected for hyper parameter tuning.
## non-natives: 12/20 4/20
class GNN24(torch.nn.Module):
    def __init__(self):
        super(GNN24, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GCN(v_feats=11, filters=16, dropout=0.4)
        self.conv2 = GCN(v_feats=16, filters=32, dropout=0.4)
        self.conv3 = GCN(v_feats=32, filters=64, dropout=0.4)
        self.conv4 = GCN(v_feats=64, filters=128, dropout=0.4)
        self.conv5 = GCN(v_feats=128, filters=128, dropout=0.4)
        self.selfAtt1 = SelfAttention(attention_size=128, batch_first=False, heads=6, nonlin="tanh")
        self.dense1 = Dense(in_dims=128*6, out_dims=256, dropout=0.4)
        self.dense2 = Dense(in_dims=256, out_dims=128, dropout=0.4)
        self.dense3 = Dense(in_dims=128, out_dims=1, nonlin="linear")

        parameters = 0
        for parameter in self.parameters():
            parameters += 1
        print(parameters)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


## Selected for hyper parameter tuning.
class GNN25(torch.nn.Module):
    def __init__(self):
        super(GNN25, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.4, multi_head=6)
        self.conv2 = GAT(v_feats=16*6, filters=32, dropout=0.4, multi_head=6)
        self.conv3 = GAT(v_feats=32*6, filters=64, dropout=0.4, multi_head=6)
        self.dense1 = Dense(in_dims=6*64, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)

# Scores: 12 5
class GNN26(torch.nn.Module):
    def __init__(self):
        super(GNN26, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = DGCN(v_feats=11, filters=64, dropout=0.4)
        self.conv2 = DGCN(v_feats=64, filters=128, dropout=0.4)
        self.conv3 = DGCN(v_feats=128, filters=256, dropout=0.4)
        self.conv4 = DGCN(v_feats=256, filters=512, dropout=0.4)
        self.selfAtt1_int = SelfAttention(attention_size=512, batch_first=False, heads=6, nonlin="tanh")
        self.selfAtt1_nh = SelfAttention(attention_size=512, batch_first=False, heads=6, nonlin="tanh")
        self.dense1 = Dense(in_dims=512*2*6, out_dims=1, nonlin="linear")

        parameters = 0
        for parameter in self.parameters():
            parameters += 1
        print("Number of parameters: ", parameters)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x_int, x_nh = x[0], x[1]

        x_int, _ = self.selfAtt1_int(x_int)
        x_nh, _ = self.selfAtt1_nh(x_nh)

        x = torch.cat((x_int, x_nh), dim=1)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


class GNN27(torch.nn.Module):
    def __init__(self):
        super(GNN27, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 2

        self.conv1 = DGCN(v_feats=11, filters=16, dropout=0.0)
        self.conv2 = DGCN(v_feats=16, filters=32, dropout=0.0)
        self.selfAtt1 = SelfAttention(attention_size=32*2, batch_first=False, heads=3, nonlin="tanh")
        self.dense1 = Dense(in_dims=32*2*3, out_dims=1, nonlin="linear")

        parameters = 0
        for parameter in self.parameters():
            parameters += 1
        print("Number of parameters: ", parameters)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_int, x_nh = x[0], x[1]

        x = torch.cat((x_int, x_nh), dim=1)

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


# non-natives: 13/20
# Hyper-parameter search.
class GNN28(torch.nn.Module):
    def __init__(self):
        super(GNN28, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.4, multi_head=6)
        self.conv2 = GAT(v_feats=16*6, filters=32, dropout=0.4, multi_head=6)
        self.conv3 = GAT(v_feats=32*6, filters=64, dropout=0.4, multi_head=6)
        self.dense1 = Dense(in_dims=6*64, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


# scores: 11
class GNN29(torch.nn.Module):
    def __init__(self):
        super(GNN29, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=32, dropout=0.4, multi_head=6)
        self.conv2 = GAT(v_feats=32*6, filters=64, dropout=0.4, multi_head=6)
        self.selfAtt1 = SelfAttention(attention_size=6*64, batch_first=False, heads=6, nonlin="tanh")
        self.dense1 = Dense(in_dims=6*6*64, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        x, _ = self.selfAtt1(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


# Scores: Currently running
# 14 14
class GNN30(torch.nn.Module):
    def __init__(self):
        super(GNN30, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.4, multi_head=6)
        self.conv2 = GAT(v_feats=16*6, filters=32, dropout=0.4, multi_head=6)
        self.conv3 = GAT(v_feats=32*6, filters=64, dropout=0.4, multi_head=6)
        self.dense1 = Dense(in_dims=6*64, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)

## Model selection: GAT, L1-loss, one label
class GNN31(torch.nn.Module):
    def __init__(self, lr=0.0001, dropout=0.4, weight_decay=0.01):
        super(GNN31, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = GAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = GAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.conv3 = GAT(v_feats=32*6, filters=64, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*64, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)

## List of hyper-parameters: dropout, lr, optimizer, weight-decay,

class GNN32(torch.nn.Module):
    def __init__(self):
        super(GNN32, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1

        self.conv1 = DGAT(v_feats=11, filters=16, multi_head=6, dropout=0.4)
        self.conv2 = DGAT(v_feats=16*6, filters=32, multi_head=6, dropout=0.4)
        self.conv3 = DGAT(v_feats=32*6, filters=64, multi_head=6, dropout=0.4)
        self.dense1 = Dense(in_dims=64*2*6, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
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

## Model selection: DGAT, L1-loss, one label
class GNN33(torch.nn.Module):
    def __init__(self, lr=0.0001, dropout=0.4, weight_decay=0.01):
        super(GNN33, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1
        self.multi_label = False

        self.conv1 = DGAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = DGAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.conv3 = DGAT(v_feats=32*6, filters=64, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*2*64, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
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


# GNN31: MS: 14 19, PS: 14, 19
class GNN34(torch.nn.Module):
    def __init__(self):
        super(GNN34, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1
        self.multi_label = False

        self.conv1 = GAT(v_feats=11, filters=16, dropout=0.1, multi_head=6)
        self.conv2 = GAT(v_feats=16*6, filters=32, dropout=0.1, multi_head=6)
        self.conv3 = GAT(v_feats=32*6, filters=64, dropout=0.1, multi_head=6)
        self.dense1 = Dense(in_dims=6*64, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        return F.l1_loss(output, target, reduction=reduction)


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


class GNN36(torch.nn.Module):
    def __init__(self, lr=0.0001, dropout=0.4, weight_decay=0.01):
        super(GNN36, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1
        self.multi_label = True

        self.conv1 = DGAT(v_feats=11, filters=16, dropout=0.3, multi_head=6)
        self.conv2 = DGAT(v_feats=16*6, filters=32, dropout=0.3, multi_head=6)
        self.conv3 = DGAT(v_feats=32*6, filters=64, dropout=0.3, multi_head=6)
        self.dense1 = Dense(in_dims=6*2*64, out_dims=5, nonlin="linear")

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
        return F.mse_loss(output, target, reduction=reduction)

## Model selection: DGAT, MSE-loss, multi-label
class GNN37(torch.nn.Module):
    def __init__(self, lr=0.0001, dropout=0.4, weight_decay=0.01):
        super(GNN37, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1
        self.multi_label = True

        self.conv1 = DGAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = DGAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.conv3 = DGAT(v_feats=32*6, filters=64, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*2*64, out_dims=5, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
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
        return F.mse_loss(output, target, reduction=reduction)

# GNN37: MS: 16 18, PS: 13 15
class GNN38(torch.nn.Module):
    def __init__(self):
        super(GNN38, self).__init__()
        self.ranking = False
        self.decoys_per_cat = 1
        self.multi_label = True

        lr = 0.05
        dropout = 0.5
        weight_decay = 0.01

        self.conv1 = DGAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = DGAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.conv3 = DGAT(v_feats=32*6, filters=64, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*2*64, out_dims=5, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
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
        return F.mse_loss(output, target, reduction=reduction)

## Model selection: DGAT, BatchRanking Loss, one label
class GNN39(torch.nn.Module):
    def __init__(self, lr=0.0001, dropout=0.4, weight_decay=0.01, amsgrad=False):
        super(GNN39, self).__init__()
        self.ranking = True
        self.decoys_per_cat = 2
        self.multi_label = False

        self.conv1 = DGAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = DGAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*2*32, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_int, x_nh = x[0], x[1]

        x = torch.cat((x_int, x_nh), dim=1)

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        loss = BatchRankingLoss(decoys_per_complex=10)
        return loss(output, target)

# GNN39: MS: 16 18, PS:
class GNN40(torch.nn.Module):
    def __init__(self):
        super(GNN40, self).__init__()
        self.ranking = True
        self.decoys_per_cat = 2
        self.multi_label = False

        lr = 0.00005
        dropout = 0.5
        weight_decay = 0.001
        amsgrad = False

        self.conv1 = DGAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = DGAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*2*32, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_int, x_nh = x[0], x[1]

        x = torch.cat((x_int, x_nh), dim=1)

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        loss = BatchRankingLoss(decoys_per_complex=10)
        return loss(output, target)

## Model selection: GAT, Batch Ranking loss, one-label
class GNN41(torch.nn.Module):
    def __init__(self, lr=0.0001, dropout=0.4, weight_decay=0.01, amsgrad=False):
        super(GNN41, self).__init__()
        self.ranking = True
        self.decoys_per_cat = 2
        self.multi_label = False

        self.conv1 = GAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = GAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*32, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        loss = BatchRankingLoss(decoys_per_complex=10)
        return loss(output, target)

class GNN42(torch.nn.Module):
    def __init__(self):
        super(GNN42, self).__init__()
        self.ranking = True
        self.decoys_per_cat = 2
        self.multi_label = False

        lr = 0.00005
        dropout = 0.5
        weight_decay = 0.001
        amsgrad = False

        self.conv1 = DGAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = DGAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*2*32, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_int, x_nh = x[0], x[1]

        x = torch.cat((x_int, x_nh), dim=1)

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        loss = BatchRankingLoss(decoys_per_complex=10)
        return loss(output, target)

# GNN41: MS: 15 19, PS:
class GNN43(torch.nn.Module):
    def __init__(self, lr=0.0001, dropout=0.4, weight_decay=0.01, amsgrad=False):
        super(GNN43, self).__init__()
        self.ranking = True
        self.decoys_per_cat = 2
        self.multi_label = False

        lr = 0.00005
        dropout = 0.1
        weight_decay = 0.00005
        amsgrad = True

        self.conv1 = GAT(v_feats=11, filters=16, dropout=dropout, multi_head=6)
        self.conv2 = GAT(v_feats=16*6, filters=32, dropout=dropout, multi_head=6)
        self.dense1 = Dense(in_dims=6*32, out_dims=1, nonlin="linear")

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[0]

        # Global Average Pooling
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)

        x = self.dense1(x)
        x = torch.squeeze(x, 1)

        return x

    def loss(self, output, target, reduction='mean'):
        loss = BatchRankingLoss(decoys_per_complex=10)
        return loss(output, target)
