import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from util import make_noise


class netG(nn.Module):
    def __init__(self, ngpu=1, device=False, action_size=31, sequence_length=10, noise_size=16, batch_size=32, noise_std=1, embed_dim=16, n_layers=1, dropout=0.2, hidden_size=64, l1_size=128):
        super(netG, self).__init__()
        self.device = device

        self.ngpu = ngpu
        self.action_size = action_size
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.noise_std = noise_std
        self.hidden_size = hidden_size
        self.l1_size=l1_size
        self.sequence_length = sequence_length

        # Embed the actions
        self.action_embedding_layer = nn.Embedding(num_embeddings=self.action_size, embedding_dim=self.embed_dim)
        # LSTM for past data
        self.lstm_1 = nn.LSTM(input_size=3+self.embed_dim, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, dropout=dropout)
        self.linear_1 = nn.Linear((self.hidden_size + self.noise_size), self.l1_size)

        self.conv_block_1 = nn.Sequential(
          nn.Conv1d(in_channels=self.l1_size, out_channels=self.l1_size, kernel_size=3, padding=1, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(self.l1_size)
        )

        self.conv_block_2 = nn.Sequential(
          nn.Conv1d(in_channels=self.l1_size, out_channels=int(self.l1_size/2), kernel_size=4, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(int(self.l1_size/2))
        )
        self.conv_block_3 = nn.Sequential(
          nn.Conv1d(in_channels=int(self.l1_size/2), out_channels=3+self.action_size, kernel_size=2, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(3+self.action_size)
        )

    def forward(self, data):
        noise = make_noise((data.shape[0], data.shape[1], self.noise_size))
        embeds = self.action_embedding_layer(data[:, :, 3:].squeeze().long())
        temporal = torch.cat((data[:, :, :3], embeds), 2)
        temporal, _ = self.lstm_1(temporal)

        temporal = torch.cat((temporal, noise), 2)
        temporal = self.linear_1(temporal)
        temporal = temporal.transpose(1,2)
        #print(temporal.shape)

        temporal = self.conv_block_1(temporal)
        #print(temporal.shape)

        temporal = self.conv_block_2(temporal)
        #print(temporal.shape)

        temporal = self.conv_block_3(temporal)
        #print(temporal.shape)

        temporal = torch.sigmoid(temporal)
        return temporal.squeeze()


class netG_onehot(nn.Module):
    def __init__(self, ngpu=1, device=False, action_size=31, sequence_length=10, noise_size=16, batch_size=32, noise_std=1, embed_dim=16, n_layers=1, dropout=0.2, hidden_size=64, l1_size=128):
        super(netG_onehot, self).__init__()
        self.device = device

        self.ngpu = ngpu
        self.action_size = action_size
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.noise_std = noise_std
        self.hidden_size = hidden_size
        self.l1_size=l1_size
        self.sequence_length = sequence_length

        # LSTM for past data
        self.lstm_1 = nn.LSTM(input_size=3+self.action_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, dropout=dropout)
        self.linear_1 = nn.Linear((self.hidden_size + self.noise_size), self.l1_size)

        self.conv_block_1 = nn.Sequential(
          nn.Conv1d(in_channels=self.l1_size, out_channels=self.l1_size, kernel_size=3, padding=1, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(self.l1_size)
        )

        self.conv_block_2 = nn.Sequential(
          nn.Conv1d(in_channels=self.l1_size, out_channels=int(self.l1_size/2), kernel_size=4, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(int(self.l1_size/2))
        )

        self.conv_block_3 = nn.Sequential(
          nn.Conv1d(in_channels=int(self.l1_size/2), out_channels=3+self.action_size, kernel_size=7, stride=1),
          nn.ReLU(),
          nn.BatchNorm1d(3+self.action_size)
        )

    def forward(self, data):
        noise = make_noise((data.shape[0], data.shape[1], self.noise_size)).to(self.device)
        temporal, _ = self.lstm_1(data.float())

        temporal = torch.cat((temporal, noise), 2)
        temporal = self.linear_1(temporal)
        temporal = temporal.transpose(1,2)
        #print(temporal.shape)
        temporal = self.conv_block_1(temporal)
        #print(temporal.shape)
        temporal = self.conv_block_2(temporal)
        #print(temporal.shape)
        temporal = self.conv_block_3(temporal)
        #print(temporal.shape)

        temporal = torch.sigmoid(temporal)
        return temporal.squeeze()


class netG_linear(nn.Module):
    def __init__(self, ngpu=1, device=False, action_size=31, sequence_length=10, noise_size=16, batch_size=32, noise_std=1, embed_dim=16, n_layers=1, dropout=0.2, hidden_size=64, l1_size=128):
        super(netG_linear, self).__init__()
        self.device = device

        self.ngpu = ngpu
        self.action_size = action_size
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.noise_std = noise_std
        self.hidden_size = hidden_size
        self.l1_size=l1_size
        self.sequence_length = sequence_length

        # Embed the actions
        self.action_embedding_layer = nn.Embedding(num_embeddings=self.action_size, embedding_dim=self.embed_dim)
        # LSTM for past data
        self.lstm_1 = nn.LSTM(input_size=3+self.embed_dim, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, dropout=dropout)
        self.linear_1 = nn.Linear((self.hidden_size + self.noise_size), self.l1_size*2, bias=False)
        self.linear_2 = nn.Linear(self.l1_size*2, self.l1_size, bias=False)
        self.linear_3 = nn.Linear(self.l1_size, 3+self.action_size, bias=False)

    def forward(self, data):
        noise = make_noise((data.shape[0], self.noise_size))
        embeds = self.action_embedding_layer(data[:,:,3:].squeeze().long())
        temporal = torch.cat((data[:,:,:3], embeds), 2)
        temporal, _ = self.lstm_1(temporal)
        temporal = temporal[:,-1,]

        temporal = torch.cat((temporal, noise), 1)
        temporal = self.linear_1(temporal)
        temporal = torch.relu(temporal)
        temporal = self.linear_2(temporal)
        temporal = torch.relu(temporal)
        temporal = self.linear_3(temporal)
        temporal = torch.sigmoid(temporal)

        return temporal.squeeze()

