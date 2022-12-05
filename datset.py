import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class FacesDataset(Dataset):
    def __init__(self, df):
        self.df = df
        # self.images = np.array(list(self.df['Image'].str.split(' '))).reshape(-1, 96, 96)
        self.mean = 0
        self.std = 0
        for i in range(len(df)):
            self.mean += np.array(list(self.df['Image'].iloc[i].split(' '))).reshape(-1, 96, 96).astype(np.float)
        self.mean = self.mean / len(df)
        for i in range(len(df)):
            self.std += (np.array(list(self.df['Image'].iloc[i].split(' '))).reshape(-1, 96, 96).astype(np.float) - self.mean)**2
        self.std = (self.std / len(df))**0.5

        # self.labels = self.df.drop('Image', axis=1).to_numpy()

        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = np.array(list(self.df['Image'].iloc[idx].split(' '))).reshape(-1, 96, 96).astype(np.float)
        image = (image - self.mean) / self.std
        label = self.df.drop('Image', axis=1).to_numpy()[idx]
        return (image, label)

