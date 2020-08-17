import torch

import os
import pickle
import random

def model_near_native_ranks(model, scores, top_n=20):
    model_scores_dict = scores

    pcomplexes = list(model_scores_dict.keys())
    pcomplexes.sort()

    total_top_20_decoys = 0
    total_hq_decoys = 0

    total_complexes = 0

    near_native_ranks = []

    dock_sw_path = "/s/jawar/b/nobackup/yash/protein-ranking/data/docking_softwares"

    for pcomplex_name in pcomplexes:
        pcomplex_ranks = []

        is_valid_pcomplex = True
        for dock_sw in model_scores_dict[pcomplex_name]:
            if(dock_sw == "natives"):
                continue

            dockq_path = os.path.join(dock_sw_path, dock_sw, "dockq")
            dockq_prot_path = os.path.join(dockq_path, pcomplex_name + ".pkl")
            dockq_dict = None
            with open(dockq_prot_path, "rb") as f:
                dockq_dict = pickle.load(f)

            pcomplex_decoys = list(model_scores_dict[pcomplex_name][dock_sw].items())

            if model.ranking:
                pcomplex_decoys.sort(key=lambda tup:tup[1], reverse=False)
            else:
                pcomplex_decoys.sort(key=lambda tup:tup[1], reverse=True)

            pcomplex_decoys_dockq = [dockq_dict[decoy_name][0] for decoy_name, score in pcomplex_decoys]

            for i, value in enumerate(pcomplex_decoys):
                decoy_name, decoy_score = value
                if(pcomplex_decoys_dockq[i] > 0.65):
                    pcomplex_ranks.append((i+1, pcomplex_decoys_dockq[i]))
                    break
                if(i == len(pcomplex_decoys)-1):
                    is_valid_pcomplex = False


        if(len(pcomplex_ranks) > 0):
            pcomplex_ranks.sort(key=lambda tup:tup[1], reverse=True)
            near_native_ranks.append(pcomplex_ranks[0][0])

        if(not is_valid_pcomplex):
            total_complexes += 1

    model_ranks = [rank for rank in near_native_ranks if(rank != None)]

    top_n_present = 0
    for rank in model_ranks:
        if(rank<=top_n):
            top_n_present += 1

    return top_n_present, total_complexes


def model_native_ranks(model, scores, top_n=20):

    model_scores_dict = scores

    pcomplexes = list(model_scores_dict.keys())
    pcomplexes.sort()

    native_ranks = []

    dock_sw_path = "/s/jawar/b/nobackup/yash/protein-ranking/data/docking_softwares"

    # pcomplexes = ['1ktz', '1s1q', '1sv0', '1u0s', '1veu', '1xb2', '2dvw', '2fhz', '2r17', '3c9a', '3k74', '3viq', '4e05', '4hwi', '4je3', '4kc3', '4yjz', '5a1n', '5bv0', '5doi']

    for pcomplex_name in pcomplexes:
        pcomplex_decoys = []

        max_dockq_docksw_val = -1
        max_dockq_docksw_cat = None
        for dock_sw in model_scores_dict[pcomplex_name]:
            if(dock_sw == "natives"):
                continue

            dockq_path = os.path.join(dock_sw_path, dock_sw, "dockq")
            dockq_prot_path = os.path.join(dockq_path, pcomplex_name + ".pkl")
            dockq_dict = None
            with open(dockq_prot_path, "rb") as f:
                dockq_dict = pickle.load(f)

            decoys_scores = model_scores_dict[pcomplex_name][dock_sw].items()

            pcomplex_decoys_dockq = [dockq_dict[decoy_name][0] for decoy_name, score in decoys_scores]

            max_pcomplex_decoys_dockq = max(pcomplex_decoys_dockq)
            if(max_pcomplex_decoys_dockq > max_dockq_docksw_val):
                max_dockq_docksw_val = max_pcomplex_decoys_dockq
                max_dockq_docksw_cat = dock_sw

        if(max_dockq_docksw_val > 0.65):
            pcomplex_decoys += model_scores_dict[pcomplex_name][max_dockq_docksw_cat].items()
            pcomplex_decoys += model_scores_dict[pcomplex_name]["natives"].items()

            if model.ranking:
                pcomplex_decoys.sort(key=lambda tup:tup[1], reverse=False)
            else:
                pcomplex_decoys.sort(key=lambda tup:tup[1], reverse=True)

            for i, value in enumerate(pcomplex_decoys):
                decoy_name, decoy_dockq = value
                if(decoy_name.endswith("_b.pdb")):
                    native_ranks.append(i)
                    break

    top_20 = 0
    for native_rank in native_ranks:
        if(native_rank <= top_n):
            top_20 += 1

    return top_20, len(native_ranks)


def model_near_native_enrichment(model, scores, top_n=20):
    model_scores_dict = scores

    pcomplexes = list(model_scores_dict.keys())
    pcomplexes.sort()

    near_native_ranks = []

    dock_sw_path = "/s/jawar/b/nobackup/yash/protein-ranking/data/docking_softwares"

    enrichment = []

    for _ in range(1000):
        enrichment_pcomplex = []
        for pcomplex_name in pcomplexes:
            enrichment_dock = []
            for dock_sw in model_scores_dict[pcomplex_name]:

                model_count = 0
                random_count = 0

                if(dock_sw == "natives"):
                    continue

                dockq_path = os.path.join(dock_sw_path, dock_sw, "dockq")
                dockq_prot_path = os.path.join(dockq_path, pcomplex_name + ".pkl")
                dockq_dict = None
                with open(dockq_prot_path, "rb") as f:
                    dockq_dict = pickle.load(f)

                pcomplex_decoys = list(model_scores_dict[pcomplex_name][dock_sw].items())

                pcomplex_decoys_len = len(pcomplex_decoys)

                pcomplex_decoys_dockq = [dockq_dict[decoy_name][0] for decoy_name, score in pcomplex_decoys]

                for i in range(len(pcomplex_decoys)):
                    decoy_name = pcomplex_decoys[i][0]
                    pcomplex_decoys[i] = list(pcomplex_decoys[i])
                    pcomplex_decoys[i].append(dockq_dict[decoy_name][0])

                pcomplex_decoys_rand = list(pcomplex_decoys)
                random.shuffle(pcomplex_decoys_rand)

                if model.ranking:
                    pcomplex_decoys.sort(key=lambda tup:tup[1], reverse=False)
                else:
                    pcomplex_decoys.sort(key=lambda tup:tup[1], reverse=True)

                good_quality_complex = False

                for i in range(pcomplex_decoys_len):
                    if(pcomplex_decoys[i][2] > 0.25):
                        good_quality_complex = True
                    if(i<=top_n and pcomplex_decoys[i][2] > 0.25):
                        model_count += 1
                    if(i<=top_n and pcomplex_decoys_rand[i][2] > 0.25):
                        random_count += 1

                if(random_count == 0):
                    random_count = 1

                if(good_quality_complex):
                    enrichment_dock.append(model_count/random_count)
            if(len(enrichment_dock)>0):
                enrichment_pcomplex.append(sum(enrichment_dock)/len(enrichment_dock))

        if(len(enrichment_pcomplex)):
            enrichment.append(sum(enrichment_pcomplex)/len(enrichment_pcomplex))

    return sum(enrichment)/len(enrichment)

def test(model, device, test_loader, epoch, two_graph_class_names, top_n=20, dataset_cat="VALID", logger=None):
    model.eval()
    test_loss = 0
    mini_batches = 0

    model_scores = {}

    with torch.no_grad():
        for batch_idx, local_batch in enumerate(test_loader):
            mini_batch_target = []
            mini_batch_output = []

            for i, item in enumerate(local_batch):

                if(item["vertices"].size()[0] == 0):
                    if(logger is not None):
                        logger.error(item["name"])
                    else:
                        print("Error: " + str(item["name"]))
                    continue

                # Move graph to GPU.
                prot_name, dock_sw, decoy_name = item["name"]
                vertices = item["vertices"].to(device)
                nh_indices = item["nh_indices"].to(device)
                int_indices = item["int_indices"].to(device)
                nh_edges = item["nh_edges"].to(device)
                int_edges = item["int_edges"].to(device)
                is_int = item["is_int"].to(device)

                model_input = None
                if(model.conv1.__class__.__name__ in two_graph_class_names):
                    model_input = (vertices, vertices, nh_indices, int_indices, nh_edges, int_edges, is_int)
                else:
                    model_input = (vertices, nh_indices, int_indices, nh_edges, int_edges, is_int)

                output = model(model_input)

                try:
                    model_scores[prot_name]
                except:
                    model_scores[prot_name] = {}

                try:
                    model_scores[prot_name][dock_sw]
                except:
                    model_scores[prot_name][dock_sw] = {}

                if(model.multi_label):
                    model_scores[prot_name][dock_sw][decoy_name] = output[0][0].item()
                else:
                    model_scores[prot_name][dock_sw][decoy_name] = output.item()

        if(dataset_cat != "CUSTOM"):
            top_n_near_native_present, total_near_native_complexes = model_near_native_ranks(model, model_scores, top_n=20)
            top_n_native_present, total_native_complexes = model_native_ranks(model, model_scores, top_n=20)
            enrichment_near_native = model_near_native_enrichment(model, model_scores, top_n=20)

            return model_scores, top_n_near_native_present, top_n_native_present, enrichment_near_native
        else:
            return model_scores
