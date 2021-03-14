"""dataset.py"""

import os
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset


class GroundTruthDataset(Dataset):
    def __init__(self, root_dir):
        # Load Ground Truth simulations from Matlab
        self.mat_data = scipy.io.loadmat(root_dir)
        
        # Load state variables
        self.z = torch.from_numpy(self.mat_data['Z']).float()

        # Extract relevant dimensions and lengths of the problem 
        self.dt = self.mat_data['dt'][0,0]
        self.dim_t = self.z.shape[0]       
        self.dim_z = self.z.shape[1]  
        self.len = self.dim_t - 1
    
    def __getitem__(self, snapshot):
        z = self.z[snapshot,:]
        return z

    def __len__(self):
        return self.len


def load_dataset(args):
    # Dataset directory path
    sys_name = args.sys_name
    root_dir = os.path.join(args.dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    dataset = GroundTruthDataset(root_dir)

    return dataset


def split_dataset(total_snaps):
    # Train and test snapshots
    train_snaps = int(0.8*total_snaps)

    # Random split
    indices = np.arange(total_snaps)
    np.random.shuffle(indices)

    train_indices = indices[:train_snaps]
    test_indices = indices[train_snaps:total_snaps]

    return train_indices, test_indices