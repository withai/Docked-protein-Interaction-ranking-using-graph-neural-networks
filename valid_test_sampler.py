import pickle
import numpy as np

from torch.utils.data import Sampler

class ValidTestSampler(Sampler):

    def __init__(self, batch_complexes, dataset_cat = "valid"):
        self.batch_complexes = batch_complexes
        self.dataset_cat = dataset_cat

        pcomplex_decoy_cat_save_path = "/s/jawar/b/nobackup/yash/protein-ranking/data/pickles/pcomplex_decoy_cat.pkl"
        self.pcomplex_decoy_cat = None

        with open(pcomplex_decoy_cat_save_path, "rb") as f:
            self.pcomplex_decoy_cat = pickle.load(f)

        self.pcomplex_names = list(self.pcomplex_decoy_cat[self.dataset_cat].keys())
        self.pcomplex_len = len(self.pcomplex_names)

    def __iter__(self):

        for batch_index in range(0, len(self.pcomplex_names), 30):
            batch = []

            if(batch_index + self.batch_complexes > self.pcomplex_len):
                batch_pcomplex_names = self.pcomplex_names[batch_index : ]
            else:
                batch_pcomplex_names = self.pcomplex_names[batch_index : batch_index + self.batch_complexes]

            for pcomplex_name in batch_pcomplex_names:

                for dock_sw in self.pcomplex_decoy_cat[self.dataset_cat][pcomplex_name]:

                    for dockq_cat in self.pcomplex_decoy_cat[self.dataset_cat][pcomplex_name][dock_sw]:

                        for decoy_name in self.pcomplex_decoy_cat[self.dataset_cat][pcomplex_name][dock_sw][dockq_cat]:

                            batch.append((pcomplex_name, dock_sw, decoy_name))

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
