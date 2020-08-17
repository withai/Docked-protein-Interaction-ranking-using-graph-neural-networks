import numpy as np
import torch
from torch.utils import data

import os
import pickle
import random

class Dataset(data.Dataset):
    def __init__(self, ranking=False, multi_label=False):
        self.device = torch.device("cuda")
        self.ranking = ranking
        self.multi_label = multi_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.complexes)

    def _get_input_feature_file_name(self, pcomplex_pick_name, pcomplex_dock_sw_pick, decoy_name_pick):
        docking_sw_directory = "/s/jawar/b/nobackup/yash/protein-ranking/data/docking_softwares/"
        directory = os.path.join(docking_sw_directory, pcomplex_dock_sw_pick, "features", "input_features_dict", pcomplex_pick_name)
        return os.path.join(directory, decoy_name_pick + ".pkl")

    def _get_dockq_scores_file_name(self, pcomplex_pick_name, pcomplex_dock_sw_pick):
        docking_sw_directory = "/s/jawar/b/nobackup/yash/protein-ranking/data/docking_softwares/"
        directory = os.path.join(docking_sw_directory, pcomplex_dock_sw_pick, "dockq")
        return os.path.join(directory, pcomplex_pick_name + ".pkl")

    def __getitem__(self, samples):
        'Generates one sample of data'

        # Select sample
        dataset = {}
        batch = []

        for sample in samples:
            pcomplex_pick_name, pcomplex_dock_sw_pick, decoy_name_pick = sample

            vertices_file = None
            nh_indices_file = None
            int_indices_file = None
            nh_edges_file = None
            int_edges_file = None
            ch_type_file = None
            is_int_file = None
            dockq_score_file = None

            input_features_dict_file = self._get_input_feature_file_name(pcomplex_pick_name, pcomplex_dock_sw_pick, decoy_name_pick[:-4])
            input_features_dict = None

            with open(input_features_dict_file, "rb") as f:
                input_features_dict = pickle.load(f)

            vertices = input_features_dict["vertices"]
            nh_indices = input_features_dict["nh_indices"]
            int_indices = input_features_dict["int_indices"]
            nh_edges = input_features_dict["nh_edges"]
            int_edges = input_features_dict["int_edges"]
            ch_type = input_features_dict["ch_type"]
            is_int = input_features_dict["is_int"]
            dockq_score_file = open(self._get_dockq_scores_file_name(pcomplex_pick_name, pcomplex_dock_sw_pick), "rb")

            data = {}
            data["name"] = (pcomplex_pick_name, pcomplex_dock_sw_pick, decoy_name_pick)

            data["vertices"] = torch.from_numpy(vertices).type(torch.double)
            data["nh_indices"] = torch.from_numpy(nh_indices).type(torch.long)
            data["int_indices"] = torch.from_numpy(int_indices).type(torch.long)
            data["nh_edges"] = torch.from_numpy(nh_edges).type(torch.double)
            data["int_edges"] = torch.from_numpy(int_edges).type(torch.double)
            data["is_int"] = torch.from_numpy(is_int).type(torch.uint8)

            data["dockq_score"] = pickle.load(dockq_score_file)[decoy_name_pick]

            if(self.multi_label):
                data["dockq_score"] = torch.from_numpy(np.array(data["dockq_score"])).type(torch.double)
                if(torch.isinf(data["dockq_score"]).any()):
                    continue
            else:
                data["dockq_score"] = torch.from_numpy(np.array(data["dockq_score"][0])).type(torch.double)

            # print("********** Input values ************")
            # print((data["vertices"] != data["vertices"]).any())
            # print((data["nh_indices"] != data["nh_indices"]).any())
            # print((data["int_indices"] != data["int_indices"]).any())
            # print((data["nh_edges"] != data["nh_edges"]).any())
            # print((data["int_edges"] != data["int_edges"]).any())
            # print("************************************")

            dockq_score_file.close()

            if(self.ranking):
                try:
                    dataset[pcomplex_pick_name].append((data, data["dockq_score"]))
                except KeyError:
                    dataset[pcomplex_pick_name] = [(data, data["dockq_score"])]
            else:
                batch.append(data)

        if(self.ranking):
            for pcomplex_pick_name in dataset:
                pcomplex_decoys = dataset[pcomplex_pick_name]
                pcomplex_decoys.sort(key=lambda tup:tup[1])

                batch += [data for data, dockq_score in pcomplex_decoys]
        else:
            random.shuffle(batch)

        assert len(batch) != 0

        return batch
