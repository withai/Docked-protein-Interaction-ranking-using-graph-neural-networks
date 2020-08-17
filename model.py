import sys
import pickle
import os
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn

from dataset import Dataset
from multi_level_sampler import MultiLevelSampler
from train_sampler import TrainSampler
from valid_test_sampler import ValidTestSampler

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

    parser.add_argument("--train_complexes",
                        help="Number of train complexes per mini-batch.",
                        type=int,
                        default=5)
    parser.add_argument("--valid_test_complexes",
                        help="Number of valid and test complexes per mini-batch.",
                        type=int,
                        default=30)
    parser.add_argument("--train_workers",
                        help="Number of pytorch workers to use for training.",
                        type=int,
                        default=0)
    parser.add_argument("--valid_test_workers",
                        help="Number of pytorch workers to use for evaluating valid and test sets.",
                        type=int,
                        default=0)
    parser.add_argument("--epoch_start",
                        help="Epoch number to start from.",
                        type=int,
                        default=1)
    parser.add_argument("--epochs",
                        help="Total number of epochs.",
                        type=int,
                        default=200)
    parser.add_argument("--mini_batch_per_epoch",
                        help="Number of mini-batches per epoch.",
                        type=int,
                        default=500)
    parser.add_argument("--valid_test_per_epochs",
                        help="Evaluate valid and test sets per epochs.",
                        type=int,
                        default=1)
    parser.add_argument("--scores_path",
                        help="Path to store results of the model from the train, valid, and test sets.",
                        type=str,
                        default="/s/jawar/b/nobackup/yash/protein-ranking/experiments/pytorch_scores")
    parser.add_argument("--load_model",
                        help="Load pre-trained model.",
                        type=bool,
                        default=False)
    parser.add_argument("--epoch_no",
                        help="Epoch number to load the pre-trained model from.",
                        type=int,
                        default=5)
    parser.add_argument("--model_path",
                        help="Path to save and load the trained model.",
                        type=str,
                        default="/s/jawar/b/nobackup/yash/protein-ranking/experiments/pytorch_models")
    parser.add_argument("--GNN_class",
                        help="GNN class containing the neural network model to train or test.",
                        type=str,
                        default="GNN34")
    parser.add_argument("--only_train",
                        help="Train or test the model.",
                        type=bool,
                        default=True)
    parser.add_argument("--patience",
                        help="The number of epochs to wait for early stopping.",
                        type=int,
                        default=15)
    parser.add_argument("--top_n",
                        help="The top-n complexes used for evaluating the dataset.",
                        type=int,
                        default=20)
    parser.add_argument("--logs_dir",
                        help="Path to store the logs.",
                        type=str,
                        default="/s/jawar/b/nobackup/yash/protein-ranking/experiments/logs")

    args = parser.parse_args()

    ##################### Set parameters #################################
    train_batch_complexes = args.train_complexes
    valid_test_batch_complexes = args.valid_test_complexes

    train_workers = args.train_workers
    valid_test_workers = args.valid_test_workers

    epoch_start = args.epoch_start
    epochs = args.epochs
    mini_batches_per_epoch = args.mini_batch_per_epoch

    valid_test_model_per_epochs = args.valid_test_per_epochs
    scores_path = args.scores_path

    load_model = args.load_model
    epoch_no = args.epoch_no
    model_dict_path = args.model_path
    gnn_class = args.GNN_class
    only_train = args.only_train

    patience = args.patience

    top_n = args.top_n

    logs_dir = args.logs_dir
    ############################################################################

    two_graph_class_names = ["DGCN", "DGAT", "EGCN"]

    # Logging.
    model_log_file = os.path.join(logs_dir, gnn_class+".log")
    logger = logging.getLogger(gnn_class)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    fh = logging.FileHandler(model_log_file)
    fh.setLevel(logging.DEBUG)

    logger.addHandler(fh)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Import GNN class.
    module = __import__("gnn")
    class_name = getattr(module, gnn_class)
    model = class_name().to(device)

    # Load model.
    if(load_model):
        logger.info("Loading model from epoch no: " + str(epoch_no))
        gnn_class_path = os.path.join(model_dict_path, gnn_class)
        if(not os.path.exists(gnn_class_path)):
            os.makedirs(gnn_class_path)
        # print(torch.load(os.path.join(gnn_class_path, str(epoch_no) + ".pth")))
        model.load_state_dict(torch.load(os.path.join(gnn_class_path, str(epoch_no) + ".pth")))

    # Model and scores save path.
    if(only_train):
        base_model_save_path = os.path.join(model_dict_path, gnn_class)

        if(not os.path.exists(base_model_save_path)):
            os.makedirs(base_model_save_path)
        else:
            os.system("rm -r " + base_model_save_path)
            os.makedirs(base_model_save_path)

    base_scores_save_path = os.path.join(scores_path, gnn_class)
    if(not os.path.exists(base_scores_save_path)):
        os.makedirs(base_scores_save_path)
    else:
        os.system("rm -r " + base_scores_save_path)
        os.makedirs(base_scores_save_path)

    logger.info(model)

    decoys_per_cat = None
    try:
        decoys_per_cat = model.decoys_per_cat
    except AttributeError:
        decoys_per_cat = 1

    # Generators
    # Training.
    params = {'sampler': TrainSampler(batch_complexes=train_batch_complexes, decoys_per_cat=decoys_per_cat, dataset_cat = "train"),
              'num_workers': train_workers,
              'collate_fn' : my_collate}
    training_set = Dataset(ranking=model.ranking, multi_label=model.multi_label)
    training_generator = data.DataLoader(training_set, **params)

    # Train dataset to test
    params = {'sampler': ValidTestSampler(batch_complexes=valid_test_batch_complexes, dataset_cat = "train"),
              'num_workers': valid_test_workers,
              'collate_fn' : my_collate}
    training_test_set = Dataset(ranking=model.ranking, multi_label=model.multi_label)
    training_test_generator = data.DataLoader(training_test_set, **params)

    # Validation.
    params = {'sampler': ValidTestSampler(batch_complexes=valid_test_batch_complexes, dataset_cat = "valid"),
              'num_workers': valid_test_workers,
              'collate_fn' : my_collate}
    validation_set = Dataset(ranking=model.ranking, multi_label=model.multi_label)
    validation_generator = data.DataLoader(validation_set, **params)

    # Test.
    params = {'sampler': ValidTestSampler(batch_complexes=valid_test_batch_complexes, dataset_cat = "test"),
              'num_workers': valid_test_workers,
              'collate_fn' : my_collate}
    test_set = Dataset(ranking=model.ranking)
    test_generator = data.DataLoader(test_set, **params)


    if(only_train):

        max_top_n = 0
        patience_count = 0

        for epoch in range(epoch_start, epochs + 1):

            train(model, device, training_generator, model.optimizer, epoch, mini_batches_per_epoch, two_graph_class_names, logger)

            if epoch % valid_test_model_per_epochs == 0:
                train_scores, train_top_n_near_native, train_top_n_native, train_enrichment_near_native = test(model, device, training_test_generator, epoch, two_graph_class_names, top_n=top_n, dataset_cat = "TRAIN")
                logger.info("Train top " + str(top_n) + ": " + str(train_top_n_near_native) + " " + str(train_top_n_native) + " " + str(train_enrichment_near_native))

                valid_scores, valid_top_n_near_native, valid_top_n_native, valid_enrichment_near_native = test(model, device, validation_generator, epoch, two_graph_class_names, top_n=top_n, dataset_cat = "VALID")
                logger.info("Validation top " + str(top_n) + ": " + str(valid_top_n_near_native) + " " + str(valid_top_n_native) + " " + str(valid_enrichment_near_native))

                test_scores, test_top_n_near_native, test_top_n_native, test_enrichment_near_native = test(model, device, test_generator, epoch_no, two_graph_class_names, top_n=top_n, dataset_cat = "TEST")
                logger.info("Test top " + str(top_n) + ": " + str(test_top_n_near_native) + " " + str(test_top_n_native) + " " + str(test_enrichment_near_native))

                if(valid_enrichment_near_native >= max_top_n):
                    max_top_n = valid_enrichment_near_native
                    patience_count = 0

                    model_save_path = base_model_save_path + "/" + str(epoch) + ".pth"
                    torch.save(model.state_dict(), model_save_path)

                    scores = {"train": train_scores, "valid": valid_scores, "test": test_scores}
                    model_save_path = base_scores_save_path + "/" + str(epoch) + ".pkl"
                    with open(model_save_path, "wb") as f:
                        pickle.dump(scores, f)
                else:
                    patience_count += 1

                if(patience_count > patience):
                    logger.info("Early stopped epoch no: " + str(epoch))
                    break
        logger.info("Training stopped epoch no:" + str(epoch))
        logger.info("Best near-native score:" + str(max_top_n))

    else:

        valid_scores, valid_top_n_near_native, valid_top_n_native, valid_enrichment_near_native  = test(model, device, validation_generator, epoch_no, two_graph_class_names, top_n=top_n, dataset_cat = "VALID")
        logger.info("Validation top " + str(top_n) + ": " + str(valid_top_n_near_native) + " " + str(valid_top_n_native) + " " + str(valid_enrichment_near_native))

        test_scores, test_top_n_near_native, test_top_n_native, test_enrichment_near_native = test(model, device, test_generator, epoch_no, two_graph_class_names, top_n=top_n, dataset_cat = "TEST")
        logger.info("Test top " + str(top_n) + ": " + str(test_top_n_near_native) + " " + str(test_top_n_native) + " " + str(test_enrichment_near_native))

        scores = {"valid": valid_scores, "test": test_scores}
        model_save_path = base_scores_save_path + "/" + str(epoch_no) + ".pkl"
        with open(model_save_path, "wb") as f:
            pickle.dump(scores, f)

if __name__ == "__main__":
    main()
