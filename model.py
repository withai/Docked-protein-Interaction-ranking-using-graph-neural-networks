import sys
import pickle
import os
import logging

import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn

from custom_dataset import customDataset
from custom_sampler import customSampler

from train import train
from test import test

torch.manual_seed(0)

def my_collate(batch):
    return batch[0]


def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Assessment of docked protein interactions using Graph Neural Networks.')

    parser.add_argument("--valid_test_complexes",
                        help="Number of valid and test complexes per mini-batch.",
                        type=int,
                        default=30)
    parser.add_argument("--valid_test_workers",
                        help="Number of pytorch workers to use for evaluating valid and test sets.",
                        type=int,
                        default=0)
    parser.add_argument("--valid_test_per_epochs",
                        help="Evaluate valid and test sets per epochs.",
                        type=int,
                        default=1)
    parser.add_argument("--dataset_path",
                        help="Path to store results of the model from the train, valid, and test sets.",
                        type=str,
                        default="")
    parser.add_argument("--epoch_no",
                        help="Epoch number to load the pre-trained model from.",
                        type=int,
                        default=5)
    parser.add_argument("--model_path",
                        help="Path to save and load the trained model.",
                        type=str,
                        default="")
    parser.add_argument("--GNN_class",
                        help="GNN class containing the neural network model to train or test.",
                        type=str,
                        default="GNN34")
    parser.add_argument("--top_n",
                        help="The top-n complexes used for evaluating the dataset.",
                        type=int,
                        default=20)
    parser.add_argument("--dataset_dir",
                        help="Path to dataset",
                        type=str,
                        default="")

    args = parser.parse_args()

    ##################### Set parameters #################################
    valid_test_batch_complexes = args.valid_test_complexes

    valid_test_workers = args.valid_test_workers

    valid_test_model_per_epochs = args.valid_test_per_epochs

    load_model = args.load_model
    epoch_no = args.epoch_no
    model_dict_path = args.model_path
    gnn_class = args.GNN_class

    top_n = args.top_n

    dataset_dir = args.dataset_dir
    ############################################################################

    two_graph_class_names = ["DGCN", "DGAT", "EGCN"]

    # Import GNN class.
    module = __import__("gnn")
    class_name = getattr(module, gnn_class)
    model = class_name().to(device)

    # Load model.
    print("Loading model from epoch no: " + str(epoch_no))
    gnn_class_path = os.path.join(model_dict_path, gnn_class)
    if(not os.path.exists(gnn_class_path)):
        os.makedirs(gnn_class_path)
    # print(torch.load(os.path.join(gnn_class_path, str(epoch_no) + ".pth")))
    model.load_state_dict(torch.load(os.path.join(gnn_class_path, str(epoch_no) + ".pth")))

    print(model)

    # Generators

    # Test.
    params = {'sampler': customSampler(batch_complexes=valid_test_batch_complexes, dataset_dir=dataset_dir),
              'num_workers': valid_test_workers,
              'collate_fn' : my_collate}
    test_set = customDataset(multi_label=model.multi_label, dataset_dir=dataset_dir)
    test_generator = data.DataLoader(test_set, **params)

    test_scores = test(model, device, test_generator, epoch_no, two_graph_class_names, top_n=top_n, dataset_cat = "CUSTOM")
    print("Scores : " + test_scores)



if __name__ == "__main__":
    main()
