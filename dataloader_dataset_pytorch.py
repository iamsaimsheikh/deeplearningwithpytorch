import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    
    def __init__(self):
        # data loading
        self.xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(self.xy[:, 1:])
        self.y = torch.from_numpy(self.xy[:, [0]])
        self.n_samples = self.xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)  # Set num_workers to 0 if issues occur

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

for epoch in range(num_epochs):
    for i, (input, labels) in enumerate(dataloader):
        # forward, backward pass
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {input.shape}, labels {labels.shape}')
