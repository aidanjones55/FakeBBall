import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd


class netD(nn.Module):
    def __init__(self, ngpu, action_size=31, sequence_length=10, noise_size=16, batch_size=32, noise_std=1, n_layers=1, dropout=0.2, hidden_size=64, l1_size=128):
        super(netD, self).__init__()

        self.ngpu = ngpu
        self.action_size = action_size
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.n_layers = n_layers
        self.noise_std = noise_std
        self.hidden_size = hidden_size
        self.l1_size=l1_size
        self.sequence_length = sequence_length

        self.lstm_1 = nn.LSTM(input_size=3+self.action_size, hidden_size=self.l1_size, num_layers=self.n_layers, batch_first=True, dropout=dropout)

        self.linear_1 = nn.Linear(self.l1_size, self.l1_size, bias=False)

        self.conv_block_1 = nn.Sequential(
          nn.Conv1d(in_channels=self.l1_size, out_channels=self.l1_size, kernel_size=4, padding=0, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(self.l1_size)
        )

        self.conv_block_2 = nn.Sequential(
          nn.Conv1d(in_channels=self.l1_size, out_channels=int(self.l1_size/2), kernel_size=4, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(int(self.l1_size/2))
        )

        self.conv_block_3 = nn.Sequential(
          nn.Conv1d(in_channels=int(self.l1_size/2), out_channels=3+self.action_size, kernel_size=5, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(3+self.action_size)
        )

        self.linear_2 = nn.Linear(3+self.action_size, 1, bias=False)

    def forward(self, data, label):
        #print(data.dtype)
        #print(label.unsqueeze(1).dtype)
        temporal = torch.cat((data, label.unsqueeze(1)), 1)
        #print(temporal.shape)
        temporal, _ = self.lstm_1(temporal.float())
        #print(temporal.shape)
        temporal = self.linear_1(temporal)
        temporal = temporal.transpose(1,2)
        #print(temporal.shape)
        temporal = self.conv_block_1(temporal)
        #print(temporal.shape)
        temporal = self.conv_block_2(temporal)
        #print(temporal.shape)
        temporal = self.conv_block_3(temporal)
        #print(temporal.shape)
        temporal = self.linear_2(temporal.squeeze())
        temporal = torch.sigmoid(temporal)
        return temporal


class netD_linear(nn.Module):
    def __init__(self, ngpu=1, action_size=31, sequence_length=10, noise_size=16, batch_size=32, noise_std=1, n_layers=1, dropout=0.2, hidden_size=64, l1_size=128):
        super(netD_linear, self).__init__()

        self.ngpu = ngpu
        self.action_size = action_size
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.n_layers = n_layers
        self.noise_std = noise_std
        self.hidden_size = hidden_size
        self.l1_size=l1_size
        self.sequence_length = sequence_length

        self.lstm_1 = nn.LSTM(input_size=3+self.action_size, hidden_size=self.l1_size, num_layers=self.n_layers, batch_first=True, dropout=dropout)

        self.linear_1 = nn.Linear(self.l1_size, int(self.l1_size / 2), bias=False)
        self.linear_2 = nn.Linear(int(self.l1_size / 2), int(self.l1_size / 4), bias=False)
        self.linear_3 = nn.Linear(int(self.l1_size / 4), 1, bias=False)

    def forward(self, data, label):
        #print(data.dtype)
        #print(label.unsqueeze(1).dtype)
        temporal = torch.cat((data, label.unsqueeze(1)), 1)
        #print(temporal.shape)
        temporal, _ = self.lstm_1(temporal.float())
        #print(temporal.shape)
        temporal = temporal[:,-1,]
        temporal = self.linear_1(temporal)
        temporal = torch.relu(temporal)
        temporal = self.linear_2(temporal)
        temporal = torch.relu(temporal)
        temporal = self.linear_3(temporal)
        temporal = torch.sigmoid(temporal)
        return temporal