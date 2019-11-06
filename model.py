import torch
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import sys
import os
from os import listdir
from os.path import isfile, join
import re
import pickle as cp
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import Dataset
from nn import GAT, GCN, bGCN, Dense

#torch.multiprocessing.set_sharing_strategy('file_system')
torch.manual_seed(0)

def my_collate(batch):
    return batch

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = bGCN(v_feats=11, filters=16, dropout=0.2)
        self.conv2 = bGCN(v_feats=16, filters=32, dropout=0.2)
        self.conv3 = bGCN(v_feats=32, filters=64, dropout=0.2)
        # self.conv4 = GAT(v_feats=64, filters=128, dropout=0.2)
        # self.conv5 = GAT(v_feats=128, filters=128, dropout=0.2)
        self.dense1 = Dense(in_dims=64, out_dims=128, dropout=0.2)
        self.dense2 = Dense(in_dims=128, out_dims=64, dropout=0.2)
        self.dense3 = Dense(in_dims=64, out_dims=1, nonlin="linear")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = x[0]
        x = torch.sum(x, 0).view(1, -1)
        x = F.normalize(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.squeeze(x, 1)

        return x

def organize_data(complex_set, type="TRAIN"):
    select_decoys_file = "/s/jawar/m/nobackup/yash/protein-ranking/data/decoys_bm4_zd3.0.2_6deg/pcomplex_select_decoys.pkl"
    f = open(select_decoys_file, "rb")
    select_decoys = cp.load(f)

    complexes = []
    for pcomplex_name in complex_set:
        if 0 not in select_decoys[pcomplex_name]:
            print("ERROR: True bound complex not found as part of the decoy.")
        for i in select_decoys[pcomplex_name]:
            complexes.append((pcomplex_name, i))

    pcomplex_weights = []
    pcomplex_categories = []
    dockq_categories_f = open("/s/jawar/b/nobackup/yash/protein-ranking/data/decoys_bm4_zd3.0.2_6deg/dockq_categories_1000.pkl", "rb")
    dockq_categories_dict = cp.load(dockq_categories_f)

    dockq_incorrect1_len = len(dockq_categories_dict[type.lower()]["incorrect1"])
    dockq_incorrect2_len = len(dockq_categories_dict[type.lower()]["incorrect2"])
    dockq_acceptable1_len = len(dockq_categories_dict[type.lower()]["acceptable1"])
    dockq_acceptable2_len = len(dockq_categories_dict[type.lower()]["acceptable2"])
    dockq_medium1_len = len(dockq_categories_dict[type.lower()]["medium1"])
    dockq_medium2_len = len(dockq_categories_dict[type.lower()]["medium2"])
    dockq_high_len = len(dockq_categories_dict[type.lower()]["high"])

    dockq_total = dockq_incorrect1_len + dockq_incorrect2_len + dockq_acceptable1_len + dockq_acceptable2_len + dockq_medium1_len + dockq_medium2_len + dockq_high_len

    weight_incorrect1 = dockq_total/dockq_incorrect1_len
    weight_incorrect2 = dockq_total/dockq_incorrect2_len
    weight_acceptable1 = dockq_total/dockq_acceptable1_len
    weight_acceptable2 = dockq_total/dockq_acceptable2_len
    weight_medium1 = dockq_total/dockq_medium1_len
    weight_medium2 = dockq_total/dockq_medium2_len
    weight_high = dockq_total/dockq_high_len

    for complex_ in complexes:
        try:
            dockq_categories_dict[type.lower()]["incorrect1"][complex_]
            pcomplex_weights.append(weight_incorrect1)
            pcomplex_categories.append(1)
        except KeyError:
            pass

        try:
            dockq_categories_dict[type.lower()]["incorrect2"][complex_]
            pcomplex_weights.append(weight_incorrect2)
            pcomplex_categories.append(2)
        except KeyError:
            pass

        try:
            dockq_categories_dict[type.lower()]["acceptable1"][complex_]
            pcomplex_weights.append(weight_acceptable1)
            pcomplex_categories.append(3)
        except KeyError:
            pass

        try:
            dockq_categories_dict[type.lower()]["acceptable2"][complex_]
            pcomplex_weights.append(weight_acceptable2)
            pcomplex_categories.append(4)
        except KeyError:
            pass

        try:
            dockq_categories_dict[type.lower()]["medium1"][complex_]
            pcomplex_weights.append(weight_medium1)
            pcomplex_categories.append(5)
        except KeyError:
            pass

        try:
            dockq_categories_dict[type.lower()]["medium2"][complex_]
            pcomplex_weights.append(weight_medium2)
            pcomplex_categories.append(6)
        except KeyError:
            pass

        try:
            dockq_categories_dict[type.lower()]["high"][complex_]
            pcomplex_weights.append(weight_high)
            pcomplex_categories.append(7)
        except KeyError:
            pass

    return complexes, pcomplex_weights, pcomplex_categories


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, local_batch in enumerate(train_loader):
        target = []
        mini_batch_output = []

        # ####################
        # categories = []
        # for item in local_batch:
        #     categories.append((item["category"], item["dockq_score"]))
        # #print(categories)
        # ####################

        for item in local_batch:
            # Move graph to GPU.
            vertices = item["vertices"].to(device)
            nh_indices = item["nh_indices"].to(device)
            int_indices = item["int_indices"].to(device)
            nh_edges = item["nh_edges"].to(device)
            int_edges = item["int_edges"].to(device)
            scores = item["dockq_score"].to(device)
            target.append(scores)

            model_input = (vertices, nh_indices, int_indices, nh_edges, int_edges)
            output = model(model_input)
            mini_batch_output.append(output)

        output = torch.stack(mini_batch_output)
        target = torch.stack(target).view(-1, 1)

        optimizer.zero_grad()

        loss = F.l1_loss(output, target, reduction='mean')
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    model_outputs = []
    decoy_order = []
    dockq_scores = []

    with torch.no_grad():
        for batch_idx, local_batch in enumerate(test_loader):
            target = []
            mini_batch_output = []
            for item in local_batch:
                # Move graph to GPU.
                prot_name, complex_no = item["name"]
                vertices = item["vertices"].to(device)
                nh_indices = item["nh_indices"].to(device)
                int_indices = item["int_indices"].to(device)
                nh_edges = item["nh_edges"].to(device)
                int_edges = item["int_edges"].to(device)
                scores = item["dockq_score"].to(device)
                target.append(scores)

                model_input = (vertices, nh_indices, int_indices, nh_edges, int_edges)
                output = model(model_input)

                model_outputs.append(output.item())
                decoy_order.append(complex_no)
                dockq_scores.append(scores.item())
                # print(output)
                # print(scores)
                mini_batch_output.append(output)

            output = torch.stack(mini_batch_output)
            target = torch.stack(target).view(-1, 1)

            test_loss += F.l1_loss(output, target, reduction='mean').item()


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return model_outputs, decoy_order, dockq_scores


def find_rank(model, device, type_="VALID"):

    train_valid_test_file = "/s/jawar/b/nobackup/yash/protein-ranking/data/decoys_bm4_zd3.0.2_6deg/train_valid_test.pkl"

    f = open(train_valid_test_file, "rb")
    train_valid_test_dict = cp.load(f)
    valid_complexes = train_valid_test_dict[type_.lower()]

    select_decoys_file = "/s/jawar/m/nobackup/yash/protein-ranking/data/decoys_bm4_zd3.0.2_6deg/pcomplex_select_decoys.pkl"
    f = open(select_decoys_file, "rb")
    select_decoys = cp.load(f)

    dockq_categories_f = open("/s/jawar/b/nobackup/yash/protein-ranking/data/decoys_bm4_zd3.0.2_6deg/dockq_categories_1000.pkl", "rb")
    dockq_categories_dict = cp.load(dockq_categories_f)

    all_complexes_perfect_decoys_predicted = 0
    all_complexes_perfect_decoys_zdock = 0
    all_complexes_perfect_decoys_dockq = 0

    for pcomplex_name in valid_complexes:

        perfect_decoys_predicted = 0
        perfect_decoys_zdock = 0

        complexes = []

        dockq_medium2_decoys = [decoy_no for complex_name, decoy_no in dockq_categories_dict[type_.lower()]["medium2"] if complex_name == pcomplex_name]
        dockq_high_decoys = [decoy_no for complex_name, decoy_no in dockq_categories_dict[type_.lower()]["high"] if complex_name == pcomplex_name]

        rank_limit = len(dockq_medium2_decoys) + len(dockq_high_decoys)
        print("Rank limit: " + str(rank_limit))

        if(rank_limit == 0):
            continue

        # Model predictions.
        for i in select_decoys[pcomplex_name]:
            complexes.append((pcomplex_name, i))

        params = {'batch_size': 32,
                  'num_workers': 1,
                  'collate_fn' : my_collate}
        validation_set = Dataset(complexes)
        training_generator = data.DataLoader(validation_set, **params)

        model_outputs, decoy_order, dockq_scores = test(model, device, training_generator)

        # sorting according to dockq
        dockq_scores, model_outputs, decoy_order = (list(x) for x in zip(*sorted(zip(dockq_scores, model_outputs, decoy_order), key=lambda pair: pair[0], reverse=True)))

        selected_decoy = None

        for i, score in enumerate(dockq_scores):
            if(score < 0.65):
                selected_decoy = i-1
                break

        assert selected_decoy != None

        model_rank = None
        dockq_scores_, model_outputs_, decoy_order_ = (list(x) for x in zip(*sorted(zip(dockq_scores, model_outputs, decoy_order), key=lambda pair: pair[1], reverse=True)))
        for i, order in enumerate(decoy_order_):
            if(order == decoy_order[selected_decoy]):
                model_rank = i+1

        print(pcomplex_name + "-  DOCKQ score: " + str(dockq_scores[selected_decoy]) + "  ZDOCK Decoy order: " + str(decoy_order[selected_decoy]) + "  Model output: " + str(model_rank))


def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #cudnn.benchmark = True

    # Datasets: Considering all complexes
    train_valid_test_file = "/s/jawar/b/nobackup/yash/protein-ranking/data/decoys_bm4_zd3.0.2_6deg/train_valid_test.pkl"

    f = open(train_valid_test_file, "rb")
    train_valid_test_dict = cp.load(f)
    train_complexes = train_valid_test_dict["train"]
    valid_complexes = train_valid_test_dict["valid"]
    test_complexes = train_valid_test_dict["test"]


    # Generators
    # Training.
    complexes, complex_weights, complex_categories = organize_data(train_complexes, type="TRAIN")
    params = {'batch_size': 32,
              'sampler': data.WeightedRandomSampler(complex_weights, len(complex_weights), replacement=True),
              'num_workers': 1,
              'collate_fn' : my_collate}
    training_set = Dataset(complexes, complex_categories)
    training_generator = data.DataLoader(training_set, **params)

    # Validation.
    complexes, complex_weights, complex_categories = organize_data(valid_complexes, type="VALID")
    params = {'batch_size': 32,
              'sampler': data.WeightedRandomSampler(complex_weights, len(complex_weights), replacement=True),
              'num_workers': 1,
              'collate_fn' : my_collate}
    validation_set = Dataset(complexes, complex_categories)
    validation_generator = data.DataLoader(validation_set, **params)

    # Test.
    # complexes, complex_weights = organize_data(valid_complexes, type="TEST")
    # params = {'batch_size': 14,
    #           'sampler': data.WeightedRandomSampler(complex_weights, len(complex_weights), replacement=True),
    #           'num_workers': 6,
    #           'collate_fn' : my_collate}
    # test_set = Dataset(complexes)
    # test_generator = data.DataLoader(test_set, **params)

    model = GNN().to(device)
    model.load_state_dict(torch.load("/s/jawar/b/nobackup/yash/protein-ranking/experiments/10-27-2019-14:32:35/5.pth")) #Baseline Graph Convolution.
    #model.load_state_dict(torch.load("/s/jawar/b/nobackup/yash/protein-ranking/experiments/10-24-2019-20:54:53/5.pth"))
    #model.load_state_dict(torch.load("/s/jawar/b/nobackup/yash/protein-ranking/experiments/10-01-2019-13:25:04/20.pth")) #without regularization Graph convolution
    #model.load_state_dict(torch.load("/s/jawar/b/nobackup/yash/protein-ranking/experiments/10-09-2019-21:43:11/20.pth")) #with dropout=0.1 regularization Graph convolution
    #model.load_state_dict(torch.load("/s/jawar/b/nobackup/yash/protein-ranking/experiments/10-13-2019-22:12:54/5.pth")) #with dropout=0.2 regularization Graph convolution
    #model.load_state_dict(torch.load("/s/jawar/b/nobackup/yash/protein-ranking/experiments/10-26-2019-18:06:47/5.pth")) #Graph attention.

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 100

    validation_interval = 5
    model_save_interval = 5
    evaluation_interval = 10

    base_model_save_path = "/s/jawar/b/nobackup/yash/protein-ranking/experiments/"
    now = datetime.now()
    base_model_save_path += now.strftime("%m-%d-%Y-%H:%M:%S")
    os.makedirs(base_model_save_path)

    train_args = {"log_interval": 20}
    epoch_start = 1
    curr_epoch = epoch_start

    for epoch in range(epoch_start, epochs + 1):
        curr_epoch = epoch
        train(train_args, model, device, training_generator, optimizer, epoch)
        if(epoch % validation_interval == 0):
            test(model, device, validation_generator)
        if(epoch % evaluation_interval == 0):
            find_rank(model, device, type_="VALID")
        if(epoch % model_save_interval == 0):
            model_save_path = base_model_save_path + "/" + str(epoch) + ".pth"
            torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main()
