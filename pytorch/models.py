
import torch
import torch.nn.functional as F
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class CustomerChurnModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomerChurnModel, self).__init__()

        self.input_layer = torch.nn.Linear(input_dim, 6)
        # self.h1 = torch.nn.Linear(6, 6)
        self.out = torch.nn.Linear(6, output_dim)


    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        # x = self.h1(x)
        # x = F.relu(x)
        x = self.out(x)

        outputs = F.sigmoid(x)
        return outputs