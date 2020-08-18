# Assessment of docked protein interaction ranking using graph neural networks

This repository uses Graph Neural Networks (GNN's) such as Graph Convolution Network (GCN) and Graph Attention Network (GAT) to rank the docked protein complexes in the order of near-nativity to its co-crystalized native complex.

## Dependency <br>
-  <a href=https://www.python.org/downloads/source/>Python 3.7 </a> <br>
-  <a href=https://pytorch.org/>PyTorch 1.1.0 library </a> (Deep learning library) <br>
-  <a href=https://pypi.org/project/numpy/>numpy</a> <br>

## Usage

```bash 
usage: model.py [-h]
                [--valid_test_complexes VALID_TEST_COMPLEXES]
                [--valid_test_workers VALID_TEST_WORKERS] 
                [--valid_test_per_epochs VALID_TEST_PER_EPOCHS]
                [--epoch_no EPOCH_NO] [--model_path MODEL_PATH]
                [--GNN_class GNN_CLASS]
                [--top_n TOP_N]

Assessment of docked protein interactions using Graph Neural Networks.

optional arguments:
  -h, --help            show this help message and exit
  --valid_test_complexes VALID_TEST_COMPLEXES
                        Number of valid and test complexes per mini-batch.
  --train_workers TRAIN_WORKERS
                        Number of pytorch workers to use for training.
  --valid_test_workers VALID_TEST_WORKERS
                        Number of pytorch workers to use for evaluating valid and test sets.
  --valid_test_per_epochs VALID_TEST_PER_EPOCHS
                        Evaluate valid and test sets per epochs.
  --epoch_no EPOCH_NO   Epoch number to load the pre-trained model from.
  --model_path MODEL_PATH
                        Path to save and load the trained model.
  --GNN_class GNN_CLASS
                        GNN class containing the neural network model to train or test.
  --top_n TOP_N         The top-n complexes used for evaluating the dataset.

```
