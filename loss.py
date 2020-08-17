import torch
import torch.nn as nn
from torch.autograd import Function


class BatchRankingLossFunction(Function):

    @staticmethod
    def forward(ctx, full_net_output, full_labels, gap, threshold, decoys_per_complex):
        full_net_output = torch.squeeze(full_net_output)
        full_batch_size = full_net_output.size(0)

        loss = torch.zeros(1, dtype=torch.float, device='cuda')
        ctx.dfdo = torch.zeros(full_batch_size, 1, dtype=torch.float, device='cuda')
        N = 0
        for batch_size in range(decoys_per_complex,full_batch_size,decoys_per_complex):
            for i in range(batch_size-decoys_per_complex,batch_size):
                for j in range(batch_size-decoys_per_complex,batch_size):
                    if i == j: continue
                    N += 1
                    tm_i = full_labels[i]
                    tm_j = full_labels[j]

                    if tm_i < tm_j:
                        y_ij = -1
                    else:
                        y_ij = 1

                    if torch.abs(tm_i - tm_j) > threshold:
                        example_weight = 1.0
                    else:
                        example_weight = 0.0

                    dL = example_weight * max(0, gap + y_ij * (full_net_output[i] - full_net_output[j]))
                    if dL > 0:
                        ctx.dfdo[i] += example_weight * y_ij
                        ctx.dfdo[j] -= example_weight * y_ij

                    loss[0] += dL

        loss /= float(N)
        ctx.dfdo /= float(N)

        return loss

    @staticmethod
    def backward(ctx, input):
        return ctx.dfdo, None, None, None, None


class BatchRankingLoss(nn.Module):
    def __init__(self, gap=1.0, threshold=0.1, decoys_per_complex=3):
        super(BatchRankingLoss, self).__init__()
        self.gap = gap
        self.threshold = threshold
        self.decoys_per_complex = decoys_per_complex

    def forward(self, input, gdt_ts):
        return BatchRankingLossFunction.apply(input, gdt_ts, self.gap, self.threshold, self.decoys_per_complex)


# if __name__ == '__main__':
#     outputs = torch.randn(10, device='cuda', dtype=torch.float32)
#     gdts = torch.randn(10, device='cuda', dtype=torch.float32)
#
#     loss = BatchRankingLoss()
#     y = loss(outputs, gdts)
#     y.backward()
#     print(y, outputs.grad)
