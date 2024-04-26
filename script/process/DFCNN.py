# -*- coding:utf-8 -*-
# @FileName : DFCNN.py
# @Time : 2024/4/16 18:47
# @Author : fiv

import torch
from torch import nn


class DFCNN(nn.Module):
    def __init__(self):
        super(DFCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 12)

    def forward(self, x):
        print(x.shape)
        x = self.relu(self.conv1(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.relu(self.conv2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.relu(self.conv3(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 2 * 2)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



if __name__ == '__main__':
    model = DFCNN()
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    print(y.size())
    print(y)