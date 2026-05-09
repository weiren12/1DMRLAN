import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as rng
import psutil
import time

class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, strides=1):
        super(ResBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            stride=strides,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity)
        return x

## Multihead
class MultiHead(nn.Module):
    def __init__(self, num_classes, default_lr=0.0001):
        super().__init__()
        self.lr = default_lr
        self.conv1 = nn.Conv1d(1, 10, kernel_size=8, padding="same")
        self.conv2 = nn.Conv1d(10, 20, kernel_size=16, padding="same")

        # Head 1
        self.conv1a = nn.Conv1d(20, 40, kernel_size=32, padding="same")
        self.res1a = ResBlock1D(40, 40)
        self.lstm1a = nn.LSTM(40, 40, num_layers=1, batch_first=True, bidirectional=False)
        self.mha1a = nn.MultiheadAttention(embed_dim=40, num_heads=10, batch_first=True)

        # Head 2
        self.conv1b = nn.Conv1d(20, 20, kernel_size=16, padding="same")
        self.res1b = ResBlock1D(20, 20)
        self.lstm1b = nn.LSTM(20, 20, num_layers=1, batch_first=True, bidirectional=False)
        self.mha1b = nn.MultiheadAttention(embed_dim=20, num_heads=10, batch_first=True)

        # Head 3
        self.conv1c = nn.Conv1d(20, 10, kernel_size=8, padding="same")
        self.res1c = ResBlock1D(10, 10)
        self.lstm1c = nn.LSTM(10, 10, num_layers=1, batch_first=True, bidirectional=False)
        self.mha1c = nn.MultiheadAttention(embed_dim=10, num_heads=10, batch_first=True)

        # Concat
        self.pool = nn.MaxPool1d(2, padding=0, stride=2)
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Head 1
        x1 = self.pool(F.relu(self.conv1a(x)))
        x1 = self.pool(F.relu(self.res1a(x1)))
        x1 = x1.transpose(2, 1).contiguous()  # change from (B,C,T) to (T,B,C), contiguous() clean memory
        outLSTM1a, _ = self.lstm1a(x1)
        outMHA1a, _ = self.mha1a(outLSTM1a, outLSTM1a, outLSTM1a)
        x1 = outMHA1a.transpose(2, 1).contiguous()

        # Head 2
        x2 = self.pool(F.relu(self.conv1b(x)))
        x2 = self.pool(F.relu(self.res1b(x2)))
        x2 = x2.transpose(2, 1).contiguous()
        outLSTM1b, _ = self.lstm1b(x2)
        outMHA1b, _ = self.mha1b(outLSTM1b, outLSTM1b, outLSTM1b)
        x2 = outMHA1b.transpose(2, 1).contiguous()

        # Head 3
        x3 = self.pool(F.relu(self.conv1c(x)))
        x3 = self.pool(F.relu(self.res1c(x3)))
        x3 = x3.transpose(2, 1).contiguous()
        outLSTM1c, _ = self.lstm1c(x3)
        outMHA1c, _ = self.mha1c(outLSTM1c, outLSTM1c, outLSTM1c)
        x3 = outMHA1c.transpose(2, 1).contiguous()

        x4 = torch.cat((x1, x2, x3), dim=1)
        x = self.pool(x4)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

