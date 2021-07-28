import os
import pickle
import numpy as np
import torch as th
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from torch.utils.data import Dataset, DataLoader, Sampler


class LightDarkDataset(Dataset):
    """
    Get a train/test dataset according to the specified settings.
    """
    def __init__(self, opt: Settings, dataset: Dict, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['observation'])

    def __getitem__(self, index):
        observation = self.dataset['observation'][index]
        action = self.dataset['action'][index]
        reward = self.dataset['reward'][index]
        next_state = self.dataset['next_state'][index]
        sample = {'observation': observation,
                  'action': action,
                  'reward': reward,
                  'next_state': next_state}
        
        if self.transform:
            sample = self.transform(sample)

        return sample


class TimeStepSampler(Sampler):
    '''
    Sampling trajectories based on #timesteps instead of #trajectories.
    '''
    def __init__(self, dataset: Dict):
        self.dataset = dataset
        self.p_sample = dataset['traj_lens']/sum(dataset['traj_lens'])
    
    def __len__(self):
        return len(self.dataset['observation'])
    
    def __iter__(self):
        # |TODO| how to use __len__()?
        index = np.random.choice(np.arange(len(self.dataset['observation'])), replace=True, p=self.p_sample)
        yield index


def get_batch(opt, data):
    batch_size = opt.batch_size
    device = opt.device
    max_len = opt.max_len
    seq_len = opt.seq_len
    dim_observation = opt.dim_observation
    dim_action = opt.dim_action
    dim_state = opt.dim_state
    dim_reward = opt.dim_reward

    o, a, r, next_s, timesteps, mask = [], [], [], [], [], [], []
    for traj in data:
        i = np.random.randint(0, len(traj['observation']) - 1)

        # get sequences from dataset
        o.append(traj['observation'][i:i + seq_len].reshape(1, -1, dim_observation))
        a.append(traj['action'][i:i + seq_len].reshape(1, -1, dim_action))
        r.append(traj['reward'][i:i + seq_len].reshape(1, -1, dim_reward))
        next_s.append(traj['next_state'][i:i + seq_len].reshape(1, -1, dim_state))
        timesteps.append(np.arange(i, i + o[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_len - 1  # padding cutoff

        # padding
        # |FIXME| check padded value & need normalization?
        tlen = o[-1].shape[1]
        o[-1] = np.concatenate([np.zeros((1, seq_len - tlen, dim_observation)), o[-1]], axis=1)
        a[-1] = np.concatenate([np.ones((1, seq_len - tlen, dim_action)) * -100., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, seq_len - tlen, 1)), r[-1]], axis=1)
        next_s[-1] = np.concatenate([np.zeros((1, seq_len - tlen, dim_state)), next_s[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, seq_len - tlen)), np.ones((1, tlen))], axis=1))

    o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=device)
    a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=device)
    r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=device)
    next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=device)
    timesteps = th.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=th.long, device=device)
    mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
    
    out = {'observation': o,
           'action': a,
           'reward': r,
           'next_state': next_s,
           'timesteps': timesteps,
           'mask': mask}

    return out

def get_loader(opt: Settings, dataset: Dict,
               transform=None, collate_fn=get_batch):
    dataset = LightDarkDataset(opt, dataset, transform, collate_fn)
    loader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=opt.shuffle,
                        collate_fn=collate_fn)
    return loader