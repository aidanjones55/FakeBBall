import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd


class CriticLinearPolicyGrad(nn.Module):
    def __init__(self, ngpu=1, device=False, action_size=31, sequence_length=10, batch_size=32, noise_std=1, embed_dim=16, n_layers=1, dropout=0.2, hidden_size=64, l1_size=64):
        super(CriticLinearPolicyGrad, self).__init__()
        self.device = device

        self.ngpu = ngpu
        self.action_size = action_size
        self.batch_size = batch_size
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
        self.linear_1 = nn.Linear((self.hidden_size), self.l1_size*2, bias=False)
        self.linear_2 = nn.Linear(self.l1_size*2, self.l1_size, bias=False)
        self.linear_3 = nn.Linear(self.l1_size, 1, bias=False)

    def forward(self, prev, label):
        #print(prev.shape)
        #print(label.shape)
        temporal = torch.cat((prev, label.unsqueeze(1)), 1)
        embeds = self.action_embedding_layer(temporal[:, :, 3:].squeeze().long())

        temporal = torch.cat((temporal[:, :, :3], embeds), 2)
        temporal, _ = self.lstm_1(temporal)
        temporal = temporal[:,-1,]
        #print(temporal.shape)

        temporal = self.linear_1(temporal)
        temporal = torch.relu(temporal)
        temporal = self.linear_2(temporal)
        temporal = torch.relu(temporal)
        temporal = self.linear_3(temporal)
        temporal = torch.sigmoid(temporal)

        return temporal.squeeze()