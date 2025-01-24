import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class NumpyDataset(Dataset):
    def __init__(self, data_path, labels_path=None):
        self.data = np.load(data_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64) if labels_path else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = torch.tensor(self.labels[idx]) if self.labels is not None else None
        return x, y

def get_data_loaders(config):
    batch_size = config["batch_size"]

    train_dataset = NumpyDataset(config["trainX"], config["trainY"])
    val_dataset = NumpyDataset(config["validX"], config["validY"])
    test_dataset = NumpyDataset(config["testX"], config["testY"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
