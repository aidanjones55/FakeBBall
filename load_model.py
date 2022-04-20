import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle

from dataLoader import get_data_loader_embed
from config import *
from generators_PG import GeneratorConvPolicyGrad


ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

model = GeneratorConvPolicyGrad(ngpu, device=device, action_size=actions_size, batch_size=batch_size, noise_size=noise_size, noise_std=noise_std, embed_dim=embed_dim, n_layers=lstm_layers, hidden_size=hidden_size)
model.load_state_dict(torch.load(f'saved_models/generators/GeneratorConvPolicyGrad_Epoch_{0}'))
model.eval()

print(model)

name_track = 'testing'
with open(f'saved_models/tracking_data/{name_track}.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict)