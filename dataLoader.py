import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import random


class SimplePlayByPlayDataset(Dataset):
    def __init__(self, data, sequence_len=16, sample_len=90, shuff=True, ret_embed=True, ret_1h=True):
        self.data = data
        self.sample_len = sample_len
        self.action_dict = {'<PAD>': 0, '<START>': 1, '<END>': 2, 'IRRS': 3, 'miss3': 4, 'F2Foul': 5, 'miss2': 6, 'DGoalTend': 7, 'PFoul': 8, 'DefReb': 9, 'CFoul': 10, 'F1Foul': 11, 'OGoalTend': 12, 'DOG': 13, 'CPFoul': 14, 'Sub': 15, 'make3': 16, 'O3sec': 17, 'OffReb': 18, 'AFPFoul': 19, 'turnover': 20, 'SFoul': 21, 'TFoul': 22, 'make2': 23, 'ejection': 24, 'violation': 25, 'D3Sec': 26, 'makeF': 27, 'missF': 28, 'OFoul': 29, 'timeout': 30}
        self.len_list = [*range(sequence_len, self.sample_len)]
        self.len_list.reverse()
        self.sequence_len = sequence_len
        self.shuff = shuff
        if self.shuff:
          random.shuffle(self.len_list)
        self.idx_counter = self.len_list.pop()

        self.ret_embed = ret_embed
        self.ret_1h = ret_1h
        #print(f'Num Quarters = {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        action_num_data = torch.tensor(np.array(self.data[idx][self.idx_counter - self.sequence_len:self.idx_counter + 1,1:4], dtype=np.float32))
        action_word_data = torch.tensor(np.array([self.action_dict[k] for k in self.data[idx][self.idx_counter - self.sequence_len:self.idx_counter + 1, 4:].flatten().tolist()]))

        action_data = torch.cat((action_num_data, action_word_data.unsqueeze(dim=1)), dim=1)
        prev_em_data, next_em_data = action_data[:-1], action_data[-1]

        onehot_prev = np.array([self.action_dict[k] for k in self.data[idx][self.idx_counter - self.sequence_len:self.idx_counter + 1, 4:].flatten().tolist()])
        onehot_prev_zeros = np.zeros((onehot_prev.size, 31))
        onehot_prev_zeros[np.arange(onehot_prev.size),onehot_prev] = 1
        onehot_prev_zeros = torch.tensor(onehot_prev_zeros)

        label_prev = torch.tensor(np.array(self.data[idx][self.idx_counter - self.sequence_len:self.idx_counter + 1,1:4], dtype=np.float32))

        label_data = torch.cat((label_prev, onehot_prev_zeros), dim = 1)

        prev_1h_data, next_1h_data = label_data[:-1], label_data[-1]

        #print(f'IDX: {idx}, idx_counter: {self.idx_counter}')

        if idx == len(self.data) - 1:
            self.idx_counter = self.len_list.pop()
            #print(f'END OF Sequence: {self.idx_counter}, Max Seq: {self.sample_len}')

        if self.len_list == []:
            self.len_list = [*range(self.sequence_len, self.sample_len)]
            self.len_list.reverse()

        if self.shuff:
            random.shuffle(self.len_list)

        if self.ret_1h and self.ret_embed:
            return prev_em_data, prev_1h_data, next_1h_data

        if self.ret_embed:
            return prev_em_data, next_em_data

        if self.ret_1h:
            return prev_1h_data, next_1h_data


def get_data_loader_embed(csv_name, batch_size=64, seq_len=10, min_sample=100, split=True, split_ratio=0.7, train_shuffle=True, valid_shuffle=False, ret_embed=True, ret_1h=True):
    raw_csv = pd.read_csv(csv_name)
    raw_csv = raw_csv.drop('Unnamed: 0', 1)

    home = ''
    away = ''
    quarter = '1'
    pct = 0

    samples = list()

    c1 = np.array([1.0, 0.0, 1.0, 1.0, ], dtype=np.float32)
    c2 = np.array(['<START>'])
    start_sample = np.concatenate((c1, c2), axis=0, dtype=np.object)

    e1 = np.array([0.0, 0.0, 0.0, 0.0, ], dtype=np.float32)
    e2 = np.array(['<END>'])
    end = np.concatenate((e1, e2), axis=0, dtype=np.object)

    p1 = np.array([0.0, 0.0, 0.0, 0.0, ], dtype=np.float32)
    p2 = np.array(['<PAD>'])
    padding = np.concatenate((p1, p2), axis=0, dtype=np.object)

    curr_sample = list()
    curr_sample.append(start_sample)

    total_len = raw_csv.shape[0]

    print('Loading Data...')
    print('0% Done')

    for index, rows in raw_csv.iterrows():
        old_pct = pct
        pct = int(100 * ((index + 1)/ total_len))
        if (pct) % 20 == 0 and pct != old_pct:
            print(f'{pct}% Done')

        if rows['Quarter'] == quarter:
            sample_to_append = rows[['TimeRemaining', 'Duration', 'Team_away', 'Team_home']].to_numpy().astype(
                np.float16)
            action_append = rows[['Action']].to_numpy()
            sample_to_append = np.concatenate((sample_to_append, action_append), axis=0)
            curr_sample.append(sample_to_append)

        else:
            curr_sample.append(end)
            samples.append(np.stack(curr_sample, axis=0))
            curr_sample = list()
            curr_sample.append(start_sample)
            quarter = rows['Quarter']
            sample_to_append = rows[['TimeRemaining', 'Duration', 'Team_away', 'Team_home']].to_numpy().astype(
                np.float16)
            action_append = rows[['Action']].to_numpy()
            sample_to_append = np.concatenate((sample_to_append, action_append), axis=0)
            curr_sample.append(sample_to_append)

    sample_len = 0

    for i, qrt in enumerate(samples):
        #print(len(qrt))
        if len(qrt) > sample_len:
            sample_len = len(qrt)

    if sample_len < min_sample:
        print(f'Warning: Minimum sample length < requested, using {sample_len}')

    else:
        sample_len = min_sample

    #print(f'sample_len: {sample_len}')

    sl_samples = list()

    for qrt in samples:
        if len(qrt) < sample_len:
            pads = np.repeat(padding.reshape([1, -1]), sample_len - len(qrt), axis=0)
            sl_samples.append(np.concatenate((qrt, pads), axis=0))
        else:
            sl_samples.append(qrt)

    train_samples = sl_samples[:int((len(sl_samples) * split_ratio / batch_size)) * batch_size]
    valid_samples = sl_samples[int((len(sl_samples) * split_ratio / batch_size)) * batch_size:]

    if train_shuffle:
        random.shuffle(train_samples)

    if valid_shuffle:
        random.shuffle(valid_samples)

    train_dataset = SimplePlayByPlayDataset(train_samples, sequence_len=seq_len, sample_len=sample_len, shuff=train_shuffle, ret_embed=ret_embed, ret_1h=ret_1h)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)

    valid_dataset = SimplePlayByPlayDataset(valid_samples, sequence_len=seq_len, sample_len=sample_len, shuff=valid_shuffle, ret_embed=ret_embed, ret_1h=ret_1h)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=valid_shuffle)

    print('Loading Complete')
    return train_dataloader, valid_dataloader, sample_len






