import os
import pdb
import argparse
import pickle as pkl
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from torch.autograd import Variable

from six.moves.urllib.request import urlretrieve
import tarfile
import pickle
import sys
import os

from dataLoader import get_data_loader_embed
from config import *
from generators import netG_onehot
from critics import netD
from util import calc_gradient_penalty

#%matplotlib inline
#os.system('')

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

train_data, valid_data = get_data_loader_embed(csv_name='data/EMBED_2015.csv', batch_size=batch_size)

net_G = netG_onehot(ngpu, device=device, action_size=actions_size, batch_size=batch_size, noise_size=noise_size, noise_std=noise_std, embed_dim=embed_dim, n_layers=lstm_layers, hidden_size=hidden_size).to(device)
net_D = netD(ngpu, action_size=actions_size, batch_size=batch_size, n_layers=lstm_layers, hidden_size=hidden_size).to(device)

optimizerD = optim.Adam(net_D.parameters(), betas=(0, 0.9), lr=lr)
optimizerG = optim.Adam(net_G.parameters(), betas=(0, 0.9), lr=lr)

G_losses = []
D_losses = []
GP_track = []
Dx = []
Dgx1 = []
Dgx2 = []
gen_track = []
iters = 0

print('STARTING TRAINING')

for epoch in range(num_epochs):

    # net_G.train()

    print(f'Epoch: {epoch}')
    for k in range(0, 90):
        for i_batch, sample_batched in enumerate(train_data):
            # Train all real data
            real_data = [_data.to(device) for _data in sample_batched]

            net_G.eval()
            net_D.train()

            optimizerD.zero_grad()

            fake_data = net_G.forward(real_data[1]).detach()

            real_logits = net_D.forward(real_data[1], real_data[2])
            fake_logits = net_D.forward(real_data[1], fake_data)

            with torch.backends.cudnn.flags(enabled=False):
                GP = calc_gradient_penalty(BATCH_SIZE=real_data[0].shape[0], LAMBDA=LAMBDA, netC=net_D,
                                           inp=real_data[1], real_data=real_data[2], fake_data=fake_data, device=device)

            loss_c = fake_logits.mean() - real_logits.mean()
            loss = loss_c + GP

            loss.backward()
            optimizerD.step()

            D_losses.append(loss.item())
            Dx.append(real_logits.round().mean().item())
            Dgx1.append(1 - fake_logits.round().mean().item())

            GP_track.append(GP.item())

            # if True:
            if i_batch % n_critic == 0:

                net_G.train()
                # net_D.eval()

                optimizerG.zero_grad()

                fake_images = net_G.forward(real_data[1])
                # fake_images = net_G.forward(real_data[0])

                fake_logits = net_D.forward(real_data[1], fake_images)
                lossG = (1 - fake_logits.mean().view(-1))

                lossG.backward()
                optimizerG.step()

                for i in range(0, n_critic):
                    Dgx2.append(1 - fake_logits.round().mean().item())
                    G_losses.append(lossG.item())

        if k % 20 == 0:
            print('Running Validation and Generating Dist.')

            net_G.eval()
            fake_dist = {0: 0}
            for k in range(0, 79):
                for i_batch, sample_batched in enumerate(valid_data):
                    real_data = [_data.cuda() for _data in sample_batched]

                    # test_fake = net_G.forward(real_data[0])
                    test_fake = net_G.forward(real_data[1])

                    # print(test_fake.shape)
                    final = torch.argmax(test_fake[:, 3:], dim=1).tolist()
                    # print(final)
                    for act in final:
                        if act in fake_dist.keys():
                            fake_dist[act] += 1
                        else:
                            fake_dist[act] = 1

            fin_gen_track = sorted([(k, v) for k, v in fake_dist.items()], key=lambda tup: tup[1], reverse=True)

            print(f'Valid Dist. Epoch {epoch}: {fin_gen_track}')

            gen_track.append(fin_gen_track)

        print(
            f'k: {k}, DLoss: {D_losses[-1]}, GLoss: {G_losses[-1]}, GP: {GP_track[-1]}, Dx: {Dx[-1]}, Dgx1: {Dgx1[-1]}, Dgx2: {Dgx2[-1]}')








