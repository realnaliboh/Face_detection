import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from datset import FacesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn

import torch.optim as optim

def show_datapoint(image, label):
    l = label.reshape(4, 2)
    X = l[:, 0]
    Y = l[:, 1]
    plt.imshow(image.reshape(96,96), cmap='Greys')
    plt.scatter(X, Y, c='black')
    plt.show()

def custom_loss(output: torch.Tensor, label: torch.Tensor):
    loss = 0
    for i in range(0, 8, 2):
        loss += torch.linalg.norm(output[:, i:i+2] - label[:, i:i+2], dim=1)**2
    return loss.mean(dim=0) / 4

model = None
model
def main():
    train_df = pd.read_csv("training.csv")

    conds = np.where((train_df.isnull().sum(axis=0) < 100))
    train_df = train_df.iloc[:, conds[0]]
    train_df = train_df.dropna(axis=0)


    traindataset = FacesDataset(train_df.iloc[:500])
    traindataloader = DataLoader(traindataset, batch_size=128, shuffle=False, num_workers=4)
    images, labels = next(iter(traindataloader))
    #show_datapoint(images[0], labels[0])

    model = resnet18()
    model.fc = nn.Linear(in_features=512, out_features=8, bias=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    criterion = custom_loss
    optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9)

    try:
        for epoch in range(1000):
            epoch_loss = []

            for image, label in tqdm(traindataloader):

                model.float()
                output = model(image.float())

                loss = criterion(output, label.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss.append(loss.item())



            mean_loss = sum(epoch_loss)/len(epoch_loss)
            print(f'Epoch loss: {mean_loss}')
    except KeyboardInterrupt:
        torch.save(model, 'model.pth')

if __name__ == '__main__':
    main()