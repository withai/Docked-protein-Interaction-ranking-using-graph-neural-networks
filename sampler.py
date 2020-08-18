import pickle
import numpy as np
import os

from torch.utils.data import Sampler

class customSampler(Sampler):

    def __init__(self, batch_complexes, dataset_dir):
        self.batch_complexes = batch_complexes
        self.dataset_dir = dataset_dir
        self.pcomplex_names = os.listdir(self.dataset_dir)
        self.pcomplex_len = len(self.pcomplex_names)


    def __iter__(self):

        for batch_index in range(0, self.pcomplex_len, self.batch_complexes):
            batch = []

            if(batch_index + self.batch_complexes > self.pcomplex_len):
                batch_pcomplex_names = self.pcomplex_names[batch_index : ]
            else:
                batch_pcomplex_names = self.pcomplex_names[batch_index : batch_index + self.batch_complexes]

            for pcomplex_name in batch_pcomplex_names:
                pcomplex_decoys = os.listdir(os.path.join(self.dataset_dir, pcomplex_name))
                for decoy_name in pcomplex_decoys:
                    batch.append((pcomplex_name, decoy_name))
            yield batch

# def main():
#     sampled_complexes = MultiLevelSampler(30, "train")
#
#     sampled_complexes_iterator = iter(sampled_complexes)
#     next(sampled_complexes_iterator)
#     next(sampled_complexes_iterator)
#
#
# if __name__ == "__main__":
#     main()
