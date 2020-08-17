import pickle
import numpy as np

from torch.utils.data import Sampler

class TrainSampler(Sampler):

    def __init__(self, batch_complexes, decoys_per_cat=1, dataset_cat = "train"):
        self.batch_complexes = batch_complexes
        self.dataset_cat = dataset_cat
        self.decoys_per_cat = decoys_per_cat

        pcomplex_pick_prob_save_path = "/s/jawar/b/nobackup/yash/protein-ranking/data/pickles/pcomplex_pick_prob.pkl"
        pcomplex_pick_prob = None

        pcomplex_dock_sw_pick_prob_save_path = "/s/jawar/b/nobackup/yash/protein-ranking/data/pickles/pcomplex_dock_sw_pick_prob.pkl"
        self.pcomplex_dock_sw_pick_prob = None

        pcomplex_decoy_cat_save_path = "/s/jawar/b/nobackup/yash/protein-ranking/data/pickles/pcomplex_decoy_cat.pkl"
        self.pcomplex_decoy_cat = None

        with open(pcomplex_pick_prob_save_path, "rb") as f:
            pcomplex_pick_prob = pickle.load(f)

        with open(pcomplex_dock_sw_pick_prob_save_path, "rb") as f:
            self.pcomplex_dock_sw_pick_prob = pickle.load(f)

        with open(pcomplex_decoy_cat_save_path, "rb") as f:
            self.pcomplex_decoy_cat = pickle.load(f)

        self.pcomplex_pick_prob_keys = list(pcomplex_pick_prob[self.dataset_cat].keys())
        self.pcomplex_pick_prob_values = [pcomplex_pick_prob[self.dataset_cat][key] for key in self.pcomplex_pick_prob_keys]

    def get_batch(self):
        batch = []

        pcomplex_pick_names = None

        pcomplex_pick_names = np.random.choice(self.pcomplex_pick_prob_keys, self.batch_complexes, p=self.pcomplex_pick_prob_values, replace=False).tolist()

        for pcomplex_pick_name in pcomplex_pick_names:

            pcomplex_dock_sw = self.pcomplex_dock_sw_pick_prob[self.dataset_cat][pcomplex_pick_name]

            pcomplex_dock_sw_keys = list(pcomplex_dock_sw.keys())
            pcomplex_dock_sw_values = [pcomplex_dock_sw[key] for key in pcomplex_dock_sw_keys]

            pcomplex_dock_sw_pick = None

            try:
                if(pcomplex_dock_sw_values[0] == None):
                    pcomplex_dock_sw_pick = np.random.choice(pcomplex_dock_sw_keys, 1, replace=False).item(0)
                else:
                    pcomplex_dock_sw_pick = np.random.choice(pcomplex_dock_sw_keys, 1, p=pcomplex_dock_sw_values, replace=False).item(0)
            except:
                print(pcomplex_dock_sw, "Error !!!")
                print(pcomplex_dock_sw_values, "Error !!!")
                print(pcomplex_pick_name, "Error !!!")

            dockq_cat = ["incorrect", "acceptable", "med_high"]

            pcomplex_dockq_cat = self.pcomplex_decoy_cat[self.dataset_cat][pcomplex_pick_name][pcomplex_dock_sw_pick]
            pcomplex_dockq_cat_native = self.pcomplex_decoy_cat[self.dataset_cat][pcomplex_pick_name]["natives"]

            med_high_decoys = pcomplex_dockq_cat["med_high"] + pcomplex_dockq_cat_native["med_high"]
            med_high_decoy_name_picks = None
            acceptable_decoy_name_picks = None
            incorrect_decoy_name_picks = None

            # Pick from each bin

            available_med_high_decoys = len(med_high_decoys)
            if(available_med_high_decoys >= 0):
                pick = self.decoys_per_cat
                not_picked = True
                while(not_picked):
                    try:
                        med_high_decoy_name_picks = [decoy_name for decoy_name in np.random.choice(med_high_decoys, pick, replace=False).tolist()]
                        not_picked = False
                        available_med_high_decoys -= pick
                    except:
                        pick -= 1

            else:
                med_high_decoy_name_picks = []

            available_acceptable_decoys = len(pcomplex_dockq_cat["acceptable"])
            if(available_acceptable_decoys > 0):
                pick = self.decoys_per_cat
                not_picked = True
                while(not_picked):
                    try:
                        acceptable_decoy_name_picks = [decoy_name for decoy_name in np.random.choice(pcomplex_dockq_cat["acceptable"], pick, replace=False).tolist()]
                        not_picked = False
                        available_acceptable_decoys -= pick
                    except:
                        pick -= 1
            else:
                acceptable_decoy_name_picks = []

            available_incorrect_decoys = len(pcomplex_dockq_cat["incorrect"])
            if(available_incorrect_decoys > 0):
                pick = self.decoys_per_cat
                not_picked = True
                while(not_picked):
                    try:
                        incorrect_decoy_name_picks = [decoy_name for decoy_name in np.random.choice(pcomplex_dockq_cat["incorrect"], pick, replace=False).tolist()]
                        not_picked = False
                        available_incorrect_decoys -= pick
                    except:
                        pick -= 1
            else:
                incorrect_decoy_name_picks = []

            # Search and pick in different bins

            remain_med_high_picks = self.decoys_per_cat - len(med_high_decoy_name_picks)
            if(remain_med_high_picks > 0):
                if(available_acceptable_decoys > 0):
                    pick = remain_med_high_picks
                    not_picked = True
                    while(not_picked):
                        try:
                            new_acceptable_pick = np.random.choice(list(set(pcomplex_dockq_cat["acceptable"]) - set(acceptable_decoy_name_picks)), pick, replace=False).tolist()
                            acceptable_decoy_name_picks += new_acceptable_pick
                            not_picked = False
                            remain_med_high_picks -= pick
                        except:
                            pick -= 1

                remain_incorrect_decoy_names = list(set(pcomplex_dockq_cat["incorrect"]) - set(incorrect_decoy_name_picks))
                if(remain_med_high_picks > 0 and len(remain_incorrect_decoy_names) > 0):
                    pick = remain_med_high_picks
                    not_picked = True
                    while(not_picked):
                        try:
                            new_incorrect_pick = np.random.choice(remain_incorrect_decoy_names, pick, replace=False).tolist()
                            incorrect_decoy_name_picks += new_incorrect_pick
                            not_picked = False
                            remain_med_high_picks -= pick
                        except:
                            pick -= 1


            remain_acceptable_picks = self.decoys_per_cat - len(acceptable_decoy_name_picks)
            if(remain_acceptable_picks > 0):
                if(available_med_high_decoys > 0):
                    pick = remain_acceptable_picks
                    not_picked = True
                    while(not_picked):
                        try:
                            new_med_high_pick = np.random.choice(list(set(med_high_decoys) - set(med_high_decoy_name_picks)), pick, replace=False).tolist()
                            med_high_decoy_name_picks += new_med_high_pick
                            not_picked = False
                            remain_acceptable_picks -= pick
                        except:
                            pick -= 1

                remain_incorrect_decoy_names = list(set(pcomplex_dockq_cat["incorrect"]) - set(incorrect_decoy_name_picks))
                if(remain_acceptable_picks > 0 and len(remain_incorrect_decoy_names) > 0):
                    pick = remain_acceptable_picks
                    not_picked = True
                    while(not_picked):
                        try:
                            new_incorrect_pick = np.random.choice(remain_incorrect_decoy_names, pick, replace=False).tolist()
                            incorrect_decoy_name_picks += new_incorrect_pick
                            not_picked = False
                            remain_acceptable_picks -= pick
                        except:
                            pick -= 1

            # TODO: Implement incorrect bin correcting later.

            # if(len(incorrect_decoy_name_picks) == 0):
            #     if(len(med_high_decoys) > 1):
            #         new_med_high_pick = [np.random.choice(list(set(med_high_decoys) - set(med_high_decoy_name_picks)), 1).item(0)]
            #         med_high_decoy_name_picks += new_med_high_pick
            #     elif(len(pcomplex_dockq_cat["acceptable"]) > 1):
            #         new_acceptable_pick = [np.random.choice(list(set(pcomplex_dockq_cat["acceptable"]) - set(acceptable_decoy_name_picks)), 1).item(0)]
            #         acceptable_decoy_name_picks += new_acceptable_pick

            per_complex_decoys = med_high_decoy_name_picks + acceptable_decoy_name_picks + incorrect_decoy_name_picks

            per_complex_batch = []

            for per_complex_decoy in per_complex_decoys:
                per_complex_decoy_name = per_complex_decoy.split(".")[0]
                if(per_complex_decoy_name[-2:] == "_b"):
                    per_complex_batch.append((pcomplex_pick_name, "natives", per_complex_decoy))
                else:
                    per_complex_batch.append((pcomplex_pick_name, pcomplex_dock_sw_pick, per_complex_decoy))

            assert len(per_complex_batch) == self.decoys_per_cat * len(dockq_cat)
            batch += per_complex_batch
        
        return batch

    def __iter__(self):
        while(True):
            yield self.get_batch()

# def main():
#     sampled_complexes = TrainSampler(30, 2, "train")
#
#     sampled_complexes_iterator = iter(sampled_complexes)
#     next(sampled_complexes_iterator)
#     next(sampled_complexes_iterator)
#
#
# if __name__ == "__main__":
#     main()
