import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from pytorch.datasets import ChurnModellingDataset
from pytorch.models import CustomerChurnModel
from pytorch.utils import split_train_test


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default="../data/Churn_Modelling.csv",
                        help='Input path for training/test dataset')
    parser.add_argument('--input-dim', type=int, default=13,
                        help='No of features in input data')
    parser.add_argument('--output-dim', type=int, default=2,
                        help='No of outputs')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Percentage for test data to be used.')

    args = parser.parse_args()
    return args

def to_numerics(input_df):
    dataset_df = input_df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    dataset_df = pd.get_dummies(dataset_df)
    features = np.array(dataset_df.drop('Exited', axis=1))
    feature_names = dataset_df.drop('Exited', axis=1).columns.tolist()
    labels = np.array(dataset_df['Exited'])
    
    if len(features) > 1:
        features = (features - features.mean(axis=0, keepdims=True)) / features.std(axis=0, keepdims=True)
    
    return features, feature_names, labels

def train(model, epoch, train_loader, loss_fn, optimizer):

    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):        
        # Forward pass: Compute predicted y by passing x to the model
        outputs = model(data)
        
        # Compute loss
        loss = loss_fn(outputs, labels)
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, loss_fn):
    
    model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += loss_fn(outputs, target)
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Preparing the data
    input_df = pd.read_csv(args.data_dir)
    features, feature_names, labels = to_numerics(input_df)

    # Create custom dataset and split train/test sets
    dataset = ChurnModellingDataset(features, feature_names, labels)
    train_set, test_set = split_train_test(dataset, args.test_size)

    # Create loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # Create model
    model = CustomerChurnModel(args.input_dim, args.output_dim)

    # Instantiate Loss Class
    criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

    # Instantiate Optimizer Class
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train Model
    for epoch in range(int(args.epochs)):
        train(model, epoch, train_loader, criterion, optimizer)
        test(model, device, test_loader, criterion)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
