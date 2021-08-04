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
    def __init__(self, config, dataset: Dict, transform=None):
        self.config = config
        self.dataset = dataset
        self.transform = transform
        
        # for get_batch()
        self.device = config.device
        self.max_len = config.max_len
        self.seq_len = config.seq_len
        self.dim_observation = config.dim_observation
        self.dim_action = config.dim_action
        self.dim_state = config.dim_state
        self.dim_reward = config.dim_reward

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


# |FIXME| hardcode -> how to use config
def get_batch(data):
    o, a, r, next_a, next_s, timesteps, mask = [], [], [], [], [], [], []
    for traj in data:
        if len(traj['observation']) == 2:
            i = 1
        else:
            i = np.random.randint(1, len(traj['observation']) - 1)

        # get sequences from dataset
        o.append(traj['observation'][:i].reshape(1, -1, 2))
        a.append(traj['action'][:i].reshape(1, -1, 2))
        r.append(traj['reward'][:i].reshape(1, -1, 1))
        next_a.append(traj['action'][i].reshape(1, -1, 2))
        next_s.append(traj['next_state'][:i].reshape(1, -1, 2))
        timesteps.append(np.arange(0, i).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= 31] = 31 - 1  # padding cutoff

        # padding
        # |FIXME| check padded value & need normalization?
        tlen = o[-1].shape[1]
        o[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), o[-1]], axis=1)
        a[-1] = np.concatenate([np.ones((1, 31 - tlen, 2)) * -100., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, 31 - tlen, 1)), r[-1]], axis=1)
        next_s[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), next_s[-1]], axis=1)
        timesteps[-1] = np.concatenate([np.zeros((1, 31 - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

    o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device('cuda'))
    a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device('cuda'))
    r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device('cuda'))
    next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device('cuda'))
    next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device('cuda'))
    timesteps = th.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=th.long, device=th.device('cuda'))
    mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device('cuda'))
    
    out = {'observation': o,
        'action': a,
        'reward': r,
        'next_action': next_a,
        'next_state': next_s,
        'timesteps': timesteps,
        'mask': mask}

    return out

def get_loader(config, dataset: Dict,
               transform=None, collate_fn=get_batch):
    dataset = LightDarkDataset(config, dataset, transform)
    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=config.shuffle,
                        collate_fn=collate_fn)
    return loader