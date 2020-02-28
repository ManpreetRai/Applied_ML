from torch.utils.data import Dataset
import torch

# Create a custom dataset
class ChurnModellingDataset(Dataset):
    def __init__(self, features, feature_names, labels):
        assert features.shape[0] == labels.shape[0], "The lengths of features and labels do not match"

        self.feature_names = feature_names
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])
    
    def get_feature_names(self):
        return self.feature_names
