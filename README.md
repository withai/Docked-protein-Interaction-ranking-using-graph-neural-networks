# Assessment of docked protein interaction ranking using graph neural networks

This repository uses Graph Neural Networks (GNN's) such as Graph Convolution Network (GCN) and Graph Attention Network (GAT) to rank the docked protein complexes in the order of near-nativity to its co-crystalized native complex.

## Dependency <br>
-  <a href=https://www.python.org/downloads/source/>Python 3.7 </a> <br>
-  <a href=https://pytorch.org/>PyTorch 1.1.0 library </a> (Deep learning library) <br>
-  <a href=https://pypi.org/project/numpy/>numpy</a> <br>

## Usage

```bash 
usage: model.py [-h] [--train_complexes TRAIN_COMPLEXES]
                [--valid_test_complexes VALID_TEST_COMPLEXES] [--train_workers TRAIN_WORKERS]
                [--valid_test_workers VALID_TEST_WORKERS] [--epoch_start EPOCH_START]
                [--epochs EPOCHS] [--mini_batch_per_epoch MINI_BATCH_PER_EPOCH]
                [--valid_test_per_epochs VALID_TEST_PER_EPOCHS] [--scores_path SCORES_PATH]
                [--load_model LOAD_MODEL] [--epoch_no EPOCH_NO] [--model_path MODEL_PATH]
                [--GNN_class GNN_CLASS] [--only_train ONLY_TRAIN] [--patience PATIENCE]
                [--top_n TOP_N] [--logs_dir LOGS_DIR]

Assessment of docked protein interactions using Graph Neural Networks.

optional arguments:
  -h, --help            show this help message and exit
  --train_complexes TRAIN_COMPLEXES
                        Number of train complexes per mini-batch.
  --valid_test_complexes VALID_TEST_COMPLEXES
                        Number of valid and test complexes per mini-batch.
  --train_workers TRAIN_WORKERS
                        Number of pytorch workers to use for training.
  --valid_test_workers VALID_TEST_WORKERS
                        Number of pytorch workers to use for evaluating valid and test sets.
  --epoch_start EPOCH_START
                        Epoch number to start from.
  --epochs EPOCHS       Total number of epochs.
  --mini_batch_per_epoch MINI_BATCH_PER_EPOCH
                        Number of mini-batches per epoch.
  --valid_test_per_epochs VALID_TEST_PER_EPOCHS
                        Evaluate valid and test sets per epochs.
  --scores_path SCORES_PATH
                        Path to store results of the model from the train, valid, and test sets.
  --load_model LOAD_MODEL
                        Load pre-trained model.
  --epoch_no EPOCH_NO   Epoch number to load the pre-trained model from.
  --model_path MODEL_PATH
                        Path to save and load the trained model.
  --GNN_class GNN_CLASS
                        GNN class containing the neural network model to train or test.
  --only_train ONLY_TRAIN
                        Train or test the model.
  --patience PATIENCE   The number of epochs to wait for early stopping.
  --top_n TOP_N         The top-n complexes used for evaluating the dataset.
  --logs_dir LOGS_DIR   Path to store the logs.

```
