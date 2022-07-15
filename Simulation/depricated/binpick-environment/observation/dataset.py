import torch
import os

import pickle

class PCDDataset(torch.utils.data.Dataset):

    def __init__(self, dir):
        self.dir = dir
        self.annotations = os.listdir(dir)
        print(f"Dataset init: {len(self.annotations)} samples")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, i):
        item_path = os.path.join(self.dir, self.annotations[i])
        
        with open(item_path, "rb") as f:
            dict = pickle.load(f)

        assert len(dict)==5000, "The pointcloud is not 5000 points"

        pcd = torch.tensor(list(dict.keys())).float().flatten()
        label = torch.tensor(list(dict.values())).float()

        return pcd, label
    