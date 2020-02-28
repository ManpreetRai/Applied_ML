from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch


def split_train_test(dataset: Dataset, test_size: float) -> (Subset, Subset):
    test_len = int(len(dataset)*(test_size))
    train_len = len(dataset) - test_len

    if test_len > train_len:
        raise Exception(f"Invalid test size {test_size}, test_len > train_len ({test_len} > {train_len})")

    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
    return train_set, test_set