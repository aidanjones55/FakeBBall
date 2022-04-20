import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick


from dataLoader import get_data_loader_embed
from config import *
from generators_PG import GeneratorConvPolicyGrad
from util import *


def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


name_track = 'testing'
with open(f'saved_models/tracking_data/{name_track}.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

with open(f'data/final_encoding.pickle', 'rb') as f:
    true_dict = pickle.load(f)

true_dict['<PAD>'] = 0
true_dict['<START>'] = 0
true_dict['<END>'] = 0

true_dict = {k: v for k, v in sorted(true_dict.items(), key=lambda item: item[1], reverse=True)}

print(true_dict)
print(loaded_dict['gen_track'][0])

true_stats = {
    '3pt': true_dict['make3']/(true_dict['make3'] + true_dict['miss3']),
    '2pt': true_dict['make2'] / (true_dict['make2'] + true_dict['miss2']),
    'ft': true_dict['makeF'] / (true_dict['makeF'] + true_dict['missF'])
}

fake_stats = {
    '3pt': list(),
    '2pt': list(),
    'ft': list()
}

#print(loaded_dict['gen_track'][0])
for i, l in enumerate(loaded_dict['gen_track']):

    name = dict([(inv_map[k], val) for k, val in l])
    print(name)
    new_name_dict = {}
    for k in true_dict.keys():
        if k in name.keys():
            new_name_dict[k] = name[k]
        else:
            new_name_dict[k] = 0

    if new_name_dict['miss3'] == 0 and new_name_dict['make3'] == 0:
        fake_stats['3pt'].append(0)
    else:
        fake_stats['3pt'].append(new_name_dict['make3']/(new_name_dict['make3'] + new_name_dict['miss3']))

    if new_name_dict['miss2'] == 0 and new_name_dict['make2'] == 0:
        fake_stats['2pt'].append(0)
    else:
        fake_stats['2pt'].append(new_name_dict['make2'] / (new_name_dict['make2'] + new_name_dict['miss2']))

    if new_name_dict['missF'] == 0 and new_name_dict['makeF'] == 0:
        fake_stats['ft'].append(0)
    else:
        fake_stats['ft'].append(new_name_dict['makeF']/(new_name_dict['makeF'] + new_name_dict['missF']))


print(fake_stats)
fig, ax1 = plt.subplots()

ax1.plot()
ax1.plot(fake_stats['3pt'], label="Fake 3pt Pct", color='r')
ax1.plot(fake_stats['2pt'], label="Fake 2pt Pct", color='b')
ax1.plot(fake_stats['ft'], label="Fake FT Pct", color='g')

plt.axhline(y=true_stats['3pt'], color='r', linestyle='--')
plt.text(20, true_stats['3pt']+0.01, 'True 3pt Pct.', color='r')

plt.axhline(y=true_stats['2pt'], color='b', linestyle='--')
plt.text(20, true_stats['2pt']+0.01, 'True 2pt Pct.', color='b')

plt.axhline(y=true_stats['ft'], color='g', linestyle='--')
plt.text(20, true_stats['ft']+0.01, 'True FT Pct.', color='g')


ax1.set_ylabel(f'Percentage')
ax1.set_xlabel(f'Epoch')

ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend()
plt.title(f"Shot Type Percentage over Epochs")
plt.savefig(f"figures/stp.png", bbox_inches='tight')



#Distribution Plots
'''
for i, l in enumerate(loaded_dict['gen_track']):

    name = dict([(inv_map[k], val) for k, val in l])
    print(name)
    new_name_dict = {}
    for k in true_dict.keys():
        if k in name.keys():
            new_name_dict[k] = name[k]
        else:
            new_name_dict[k] = 0
    # calculate slope and intercept for the linear trend line

    fig, ax1 = plt.subplots()
    ax1.bar(new_name_dict.keys(), new_name_dict.values(), align='center', color='r')
    ax2 = ax1.twinx()
    ax2.bar(true_dict.keys(), true_dict.values(), align='edge', color='b')

    ax1.set_ylabel(f'Encoded Frequency Epoch {i}', color='r')
    ax2.set_ylabel('Encoded Frequency True', color='b')
    ax1.tick_params(axis='x', rotation=90)
    plt.title(f"Encoded Frequency, True vs. Generated Epoch {i}")
    plt.savefig(f"figures/distributions/distEpoch{i}.png", bbox_inches='tight')
'''

#Critic Accuracy MAvg Example
'''
maw = 1000

fig, ax1 = plt.subplots()

#color = 'tab:red'
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Accuracy')
ax1.plot(moving_avg(loaded_dict['Dx'], maw), label="$D(x)$")
ax1.plot(moving_avg(loaded_dict['Dgx1'], maw), label="$D(G(z)_1)$")
ax1.plot(moving_avg(loaded_dict['Dgx2'], maw), label="$D(G(z)_2)$")

ax1.legend(loc=0)

plt.title(f"Critic Accuracy During Training ({maw} Iteration MAvg)")
plt.savefig("figures/GCAccuracy1.png", bbox_inches='tight')
'''

#Generator Critic Loss
'''
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss (D)')
d = ax1.plot(loaded_dict['D_losses'], label="D", color=color)
ax1.legend(loc=0)


ax2 = ax1.twinx()

ax2.set_ylabel('Loss (G)')
g = ax2.plot(loaded_dict['G_losses'], label="G")

lns = d + g
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)


plt.title("Generator and Critic Loss During Training")
plt.savefig("figures/GCLoss1.png", bbox_inches='tight')
'''

# Sample Importing of Models
'''
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

model = GeneratorConvPolicyGrad(ngpu, device=device, action_size=actions_size, batch_size=batch_size, noise_size=noise_size, noise_std=noise_std, embed_dim=embed_dim, n_layers=lstm_layers, hidden_size=hidden_size)

epoch = 10

model.load_state_dict(torch.load(f'saved_models/generators/GeneratorConvPolicyGrad_Epoch_{epoch}'))
model.eval()

print(model)
'''