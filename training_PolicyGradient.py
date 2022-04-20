import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle

from dataLoader import get_data_loader_embed
from config import *
from generators_PG import GeneratorConvPolicyGrad
from critic_PG import CriticLinearPolicyGrad
from util import calc_gradient_penalty

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

train_data, valid_data, max_sample_length = get_data_loader_embed(csv_name='data/EMBED_2015.csv', batch_size=batch_size, seq_len=sequence_len, split_ratio=0.90, train_shuffle=True, ret_embed=True, ret_1h=False)
#train_data, valid_data, max_sample_length = get_data_loader_embed(csv_name='data/test.csv', batch_size=batch_size, seq_len=sequence_len, split_ratio=0.95, train_shuffle=True, ret_embed=True, ret_1h=False)


net_G = GeneratorConvPolicyGrad(ngpu, device=device, action_size=actions_size, batch_size=batch_size, noise_size=noise_size, noise_std=noise_std, embed_dim=embed_dim, n_layers=lstm_layers, hidden_size=hidden_size).to(device)
net_D = CriticLinearPolicyGrad(ngpu, action_size=actions_size, batch_size=batch_size, n_layers=lstm_layers, hidden_size=hidden_size).to(device)

optimizerD = optim.Adam(net_D.parameters(), betas=(0, 0.9), lr=lr)
optimizerG = optim.Adam(net_G.parameters(), betas=(0, 0.9), lr=lr)

log_dict = {
    'G_losses': [],
    'D_losses': [],
    'GP_track': [],
    'Dx': [],
    'Dgx1': [],
    'Dgx2': [],
    'gen_track': [],
    'iters': 0
}

for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    fake_dist = {0: 0}

    for k in range(0, max_sample_length - sequence_len):

        for i_batch, sample_batched in enumerate(train_data):

            # Train all real data
            real_data = [_data.to(device) for _data in sample_batched]

            '''
            
            '''

            net_G.eval()
            net_D.train()

            optimizerD.zero_grad()

            fake_data = net_G.forward(real_data[0]).detach()
            distr_time = F.normalize(fake_data[:, :25], p=1, dim=1)
            distr_home = fake_data[:, 25:27]
            distr_away = fake_data[:, 27:29]
            distr_action = F.normalize(fake_data[:, 29:], p=1, dim=1)

            one_hot_time = torch.zeros(distr_time.size(), dtype=torch.uint8).to(device)
            one_hot_home = torch.zeros(distr_home.size(), dtype=torch.uint8).to(device)
            one_hot_away = torch.zeros(distr_away.size(), dtype=torch.uint8).to(device)
            one_hot_action = torch.zeros(distr_action.size(), dtype=torch.uint8).to(device)

            time = torch.multinomial(fake_data[:, :25], num_samples=1, replacement=True) / 24
            home = torch.multinomial(fake_data[:, 25:27], num_samples=1, replacement=True)
            away = torch.multinomial(fake_data[:, 27:29], num_samples=1, replacement=True)
            action = torch.multinomial(fake_data[:, 29:], num_samples=1, replacement=True)

            fake_data_embed = torch.cat((time, home, away, action), dim=1)

            real_logits = net_D.forward(real_data[0], real_data[1])
            fake_logits = net_D.forward(real_data[0], fake_data_embed)

            with torch.backends.cudnn.flags(enabled=False):
                GP = calc_gradient_penalty(BATCH_SIZE=real_data[0].shape[0], LAMBDA=LAMBDA, netC=net_D,
                                           inp=real_data[0], real_data=real_data[1], fake_data=fake_data_embed, device=device)

            loss_c = fake_logits.mean() - real_logits.mean()
            loss = loss_c + GP

            loss.backward()
            optimizerD.step()

            log_dict['D_losses'].append(loss.item())
            log_dict['Dx'].append(real_logits.round().mean().item())
            log_dict['Dgx1'].append(1 - fake_logits.round().mean().item())
            log_dict['GP_track'].append(GP.item())

            # if True:
            if i_batch % n_critic == 0:
                net_G.train()
                #net_D.eval()

                optimizerG.zero_grad()

                fake_data = net_G.forward(real_data[0])
                distr_time = F.normalize(fake_data[:, :25], p=1, dim=1)
                distr_home = F.normalize(fake_data[:, 25:27], p=1, dim=1)
                distr_away = F.normalize(fake_data[:, 27:29], p=1, dim=1)
                distr_action = F.normalize(fake_data[:, 29:], p=1, dim=1)

                one_hot_time = torch.zeros(distr_time.size(), dtype=torch.uint8).to(device)
                one_hot_home = torch.zeros(distr_home.size(), dtype=torch.uint8).to(device)
                one_hot_away = torch.zeros(distr_away.size(), dtype=torch.uint8).to(device)
                one_hot_action = torch.zeros(distr_action.size(), dtype=torch.uint8).to(device)

                time = torch.multinomial(fake_data[:, :25], num_samples=1, replacement=True) / 24
                home = torch.multinomial(fake_data[:, 25:27], num_samples=1, replacement=True)
                away = torch.multinomial(fake_data[:, 27:29], num_samples=1, replacement=True)
                action = torch.multinomial(fake_data[:, 29:], num_samples=1, replacement=True)

                fake_data_embed = torch.cat((time, home, away, action), dim=1)
                one_hot_time.scatter_(1, (time*24).data.view(-1, 1).long(), 1)
                one_hot_home.scatter_(1, home.data.view(-1, 1).long(), 1)
                one_hot_away.scatter_(1, away.data.view(-1, 1).long(), 1)
                one_hot_action.scatter_(1, action.data.view(-1, 1).long(), 1)

                loss_time = torch.masked_select(distr_time, one_hot_time)
                loss_home = torch.masked_select(distr_home, one_hot_home)
                loss_away = torch.masked_select(distr_away, one_hot_away)
                loss_action = torch.masked_select(distr_action, one_hot_action)

                loss = loss_time*loss_home*loss_away*loss_action

                reward = ((net_D.forward(real_data[0], fake_data_embed))*2)-1
                loss = loss*reward
                loss = -torch.mean(loss)
                loss.backward()

                optimizerG.step()

                for i in range(0, n_critic):
                    log_dict['Dgx2'].append(1 - ((reward+1)/2).round().mean().item())
                    log_dict['G_losses'].append(loss.item())

                #print(action.squeeze().tolist())
                for act in action.squeeze().tolist():
                    if act in fake_dist.keys():
                        fake_dist[act] += 1
                    else:
                        fake_dist[act] = 1

        print(f'k: {k}, DLoss: {log_dict["D_losses"][-1]}, GLoss: {log_dict["G_losses"][-1]}, GP: {log_dict["GP_track"][-1]}, Dx: {log_dict["Dx"][-1]}, Dgx1: {log_dict["Dgx1"][-1]}, Dgx2: {log_dict["Dgx2"][-1]}')

    torch.save(net_G.state_dict(), f'saved_models/generators/GeneratorConvPolicyGrad_Epoch_{epoch}')

    name_track = 'testing'
    with open(f'saved_models/tracking_data/{name_track}.pkl', 'wb') as f:
        pickle.dump(log_dict, f)

    fin_gen_track = sorted([(k, v) for k, v in fake_dist.items()], key=lambda tup: tup[1], reverse=True)
    log_dict["gen_track"].append(fin_gen_track)
    print(f'Valid Dist. Epoch {epoch}: {fin_gen_track}')
