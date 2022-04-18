import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd


def make_noise(shape, volatile=False):
    tensor = torch.randn(shape)
    return autograd.Variable(tensor, volatile)


def calc_gradient_penalty(BATCH_SIZE, LAMBDA, netC, inp, real_data, fake_data, device):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netC(inp, interpolates)

    # print(disc_interpolates.shape)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

action_dict = {'<PAD>': 0, '<START>': 1, '<END>': 2, 'IRRS': 3, 'miss3': 4, 'F2Foul': 5, 'miss2': 6, 'DGoalTend': 7, 'PFoul': 8, 'DefReb': 9, 'CFoul': 10, 'F1Foul': 11, 'OGoalTend': 12, 'DOG': 13, 'CPFoul': 14, 'Sub': 15, 'make3': 16, 'O3sec': 17, 'OffReb': 18, 'AFPFoul': 19, 'turnover': 20, 'SFoul': 21, 'TFoul': 22, 'make2': 23, 'ejection': 24, 'violation': 25, 'D3Sec': 26, 'makeF': 27, 'missF': 28, 'OFoul': 29, 'timeout': 30}
