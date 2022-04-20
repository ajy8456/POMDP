import os
import glob
import pickle
import numpy as np
import torch as th
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from dataclasses import dataclass, replace
from simple_parsing import Serializable

import pdb

from wandb import set_trace


class LightDarkDataset(Dataset):
    """
    Get a train/test dataset according to the specified settings.
    """
    def __init__(self, config, dataset: List, transform=None):
        self.config = config
        self.dataset = dataset # list of data file name
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        d = self.dataset[index]
        with open(d, 'rb') as f:
            try:
                # print(self.dataset[index])
                d = pickle.load(f)
            except pickle.UnpicklingError:
                pass
        
        if self.transform:
            d = self.transform(d)

        if self.config.randomize:
            traj = d[0]
            goal_state = d[1]
        else:
            traj = d
            
        # print('========================================')
        # print(traj)
        action = np.asarray(traj[:, 0])
        observation = traj[:, 1]
        next_state = traj[:, 2]
        reward = traj[:, 3]

        data = {'action': action.tolist(),
                'observation': observation.tolist(),
                'next_state': next_state.tolist(),
                'reward': reward.tolist()}
        
        if self.config.randomize:
            data['goal_state'] = np.asarray(goal_state)

        return data

    # def __init__(self, config, dataset: Dict, transform=None):
    #     self.config = config
    #     self.dataset = dataset
    #     self.transform = transform
        
    #     # for get_batch()
    #     self.device = config.device
    #     self.max_len = config.max_len
    #     self.seq_len = config.seq_len
    #     self.dim_observation = config.dim_observation
    #     self.dim_action = config.dim_action
    #     self.dim_state = config.dim_state
    #     self.dim_reward = config.dim_reward

    #     # # for WeightedRandomSampler
    #     # self.p_sample = dataset['p_sample']

    # def __len__(self):
    #     return len(self.dataset['observation'])

    # def __getitem__(self, index):
    #     observation = self.dataset['observation'][index]
    #     action = self.dataset['action'][index]
    #     reward = self.dataset['reward'][index]
    #     next_state = self.dataset['next_state'][index]
    #     traj_len = self.dataset['traj_len'][index]
    #     goal_state = self.dataset['goal_state'][index]
    #     total_reward = self.dataset['total_reward'][index]

    #     sample = {'observation': observation,
    #               'action': action,
    #               'reward': reward,
    #               'next_state': next_state,
    #               'goal_state': goal_state,
    #               'total_reward': total_reward,
    #               'traj_len': traj_len}
        
    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample


class BatchMaker():
    def __init__(self, config):
        self.config = config
        self.device = config.device

    def __call__(self, data):
        o, a, r, next_a, next_s, next_r, goal_s, accumulated_r, timestep, mask = [], [], [], [], [], [], [], [], [], []
        for traj in data:
            if len(traj['observation']) == 2:
                i = 1
            else:
                i = np.random.randint(1, len(traj['observation']) - 1)

            # get sequences from dataset
            o.append(np.asarray(traj['observation'])[:i].reshape(1, -1, 2))
            a.append(np.asarray(traj['action'][:i]).reshape(1, -1, 2))
            r.append(np.asarray(traj['reward'][:i]).reshape(1, -1, 1))
            next_a.append(np.asarray(traj['action'])[i].reshape(1, -1, 2))
            next_r.append(np.asarray(traj['reward'])[i].reshape(1, -1, 1))
            next_s.append(np.asarray(traj['next_state'])[1:i+1].reshape(1, -1, 2))
            accumulated_r.append(np.sum(np.asarray(traj['reward'][i:])).reshape(1, -1))

            # if np.sum(np.asarray(traj['reward'][i:])) < 0.0:
            #     pdb.set_trace()

            if self.config.randomize:
                goal_s.append(traj['goal_state'].reshape(1, -1, 2))
            timestep.append(np.arange(0, i).reshape(1, -1))
            timestep[-1][timestep[-1] >= 31] = 31 - 1  # padding cutoff

            # padding
            # |FIXME| check padded value & need normalization?
            tlen = o[-1].shape[1]
            o[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), o[-1]], axis=1)
            a[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), a[-1]], axis=1)
            # a[-1] = np.concatenate([np.ones((1, 31 - tlen, 2)) * -100., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, 31 - tlen, 1)), r[-1]], axis=1)
            next_s[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), next_s[-1]], axis=1)
            timestep[-1] = np.concatenate([np.zeros((1, 31 - tlen)), timestep[-1]], axis=1)
            if self.config.randomize:
                mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1))
            else:
                mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        accumulated_r = th.from_numpy(np.concatenate(accumulated_r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        if self.config.randomize:
            goal_s = th.from_numpy(np.concatenate(goal_s, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        
        out = {'observation': o,
            'action': a,
            'reward': r,
            'next_action': next_a,
            'next_reward': next_r,
            'next_state': next_s,
            'accumulated_reward': accumulated_r,
            'timestep': timestep,
            'mask': mask}
        
        if self.config.randomize:
            out['goal_state'] = goal_s

        return out


class MultiTargetLightDarkDataset(Dataset):
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

        # # for WeightedRandomSampler
        # self.p_sample = dataset['p_sample']

    def __len__(self):
        return len(self.dataset['observation'])

    def __getitem__(self, index):
        # get sequences from dataset
        observation = self.dataset['observation'][index]
        action = self.dataset['action'][index]
        reward = self.dataset['reward'][index]
        next_state = self.dataset['next_state'][index]
        traj_len = self.dataset['traj_len'][index]
        goal_state = self.dataset['goal_state'][index]
        total_reward = self.dataset['total_reward'][index]

        traj = {'observation': observation,
                  'action': action,
                  'reward': reward,
                  'next_state': next_state,
                  'traj_len': traj_len,
                  'goal_state': goal_state,
                  'total_reward': total_reward}

        if len(traj['observation']) == 2:
            i = 1
        else:
            i = np.random.randint(1, len(traj['observation']) - 1)

        sample = self._collect_target(traj, i)
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _collect_target(self, traj, i):
        # truncate & fit interface of sample to model
        o, a, r, next_a, next_s, next_r, goal_s, total_r, timestep, mask = [], [], [], [], [], [], [], [], [], []
        o.append(traj['observation'][:i].reshape(-1, 2))
        a.append(traj['action'][:i].reshape(-1, 2))
        r.append(traj['reward'][:i].reshape(-1, 1))
        next_a.append(np.round(traj['action'][i], 4).reshape(-1, 2))
        next_r.append(traj['reward'][i].reshape(-1, 1))
        next_s.append(traj['next_state'][1:i+1].reshape(-1, 2))
        total_r.append(traj['total_reward'])
        goal_s.append(traj['goal_state'].reshape(1, -1, 2))
        timestep.append(np.arange(0, i))
        timestep[-1][timestep[-1] >= 31] = 31 - 1  # padding cutoff

        # collect multi-target indices
        # |TODO| how to use full len?
        target_index = []
        if i == 1:
            o0 = np.round(traj['observation'][0], 4)
            for idx in range(len(self.dataset['observation'])):
                if np.array_equal(np.round(self.dataset['observation'][idx][0] ,4), o0):
                    target_index.append(idx)

        elif i == 2:
            # take first action in sample
            a1 = np.round(traj['action'][1], 4)
            o1 = np.round(traj['observation'][1], 4)
            for idx in range(len(self.dataset['observation'])):
                if len(self.dataset['action'][idx]) < i+1:
                    continue
                if np.array_equal(np.round(self.dataset['action'][idx][1], 4), a1) and np.array_equal(np.round(self.dataset['observation'][idx][1], 4), o1):
                    target_index.append(idx)
        
        elif i == 3:
            a1 = np.round(traj['action'][1], 4)
            o1 = np.round(traj['observation'][1], 4)
            a2 = np.round(traj['action'][2], 4)
            o2 = np.round(traj['observation'][2], 4)
            for idx in range(len(self.dataset['observation'])):
                if len(self.dataset['action'][idx]) < i+1:
                    continue
                if np.array_equal(np.round(self.dataset['action'][idx][1], 4), a1) and np.array_equal(np.round(self.dataset['observation'][idx][1], 4), o1) and np.array_equal(np.round(self.dataset['action'][idx][2], 4), a2) and np.array_equal(np.round(self.dataset['observation'][idx][2], 4), o2):
                    target_index.append(idx)

        # Collect multi-targets
        if target_index:
            for t in target_index:
                # |FIXME| IndexError: index 3 is out of bounds for axis 0 with size 3
                if len(self.dataset['action'][t]) < i+1:
                    continue
                next_a.append(np.round(self.dataset['action'][t][i], 4).reshape(-1, 2))

        # padding
        tlen = o[-1].shape[-2]
        o[-1] = np.concatenate([np.zeros((31 - tlen, 2)), o[-1]], axis=-2)
        a[-1] = np.concatenate([np.zeros((31 - tlen, 2)), a[-1]], axis=-2)
        r[-1] = np.concatenate([np.zeros((31 - tlen, 1)), r[-1]], axis=-2)
        next_s[-1] = np.concatenate([np.zeros((31 - tlen, 2)), next_s[-1]], axis=-2)
        timestep[-1] = np.concatenate([np.zeros((31 - tlen)), timestep[-1]])
        # mask.append(np.concatenate([np.full(31 - tlen, False, dtype=bool), np.full(tlen, True, dtype=bool)]))
        mask.append(np.concatenate([np.full((1, self.seq_len - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        # next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        total_r = th.from_numpy(np.asarray(total_r).reshape(-1, 1)).to(dtype=th.float32, device=th.device(self.config.device))
        goal_s = th.from_numpy(np.concatenate(goal_s, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.config.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.config.device))

        next_a = np.array(next_a).reshape(-1, 2)
        next_a = np.unique(next_a, axis=-2)
        next_a = th.from_numpy(next_a).to(dtype=th.float32, device=th.device(self.config.device))
        
        data = {'observation': o,
            'action': a,
            'reward': r,
            'next_action': next_a,
            'next_reward': next_r,
            'next_state': next_s,
            'goal_state': goal_s,
            'total_reward': total_r,
            'timestep': timestep,
            'mask': mask}

        return data

# # |TODO| make collate_fn for multi-target
# class MultiTargetBatchMaker():
#     def __init__(self, config):
#         self.config = config
#         self.device = config.device

#     def __call__(self, data):
#         o, a, r, next_a, next_s, next_r, timestep, mask = [], [], [], [], [], [], [], []
#         for traj in data:
#             if len(traj['observation']) == 2:
#                 i = 1
#             else:
#                 i = np.random.randint(1, len(traj['observation']) - 1)
            
#             # get sequences from dataset
#             o.append(traj['observation'][:i].reshape(1, -1, 2))
#             a.append(traj['action'][:i].reshape(1, -1, 2))
#             r.append(traj['reward'][:i].reshape(1, -1, 1))
#             next_a.append(traj['action'][i].reshape(1, -1, 2))
#             next_r.append(traj['reward'][i].reshape(1, -1, 1))
#             next_s.append(traj['next_state'][1:i+1].reshape(1, -1, 2))
#             timestep.append(np.arange(0, i).reshape(1, -1))
#             timestep[-1][timestep[-1] >= 31] = 31 - 1  # padding cutoff

#             # padding
#             # |FIXME| check padded value & need normalization?
#             tlen = o[-1].shape[1]
#             o[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), o[-1]], axis=1)
#             a[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), a[-1]], axis=1)
#             # a[-1] = np.concatenate([np.ones((1, 31 - tlen, 2)) * -100., a[-1]], axis=1)
#             r[-1] = np.concatenate([np.zeros((1, 31 - tlen, 1)), r[-1]], axis=1)
#             next_s[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), next_s[-1]], axis=1)
#             timestep[-1] = np.concatenate([np.zeros((1, 31 - tlen)), timestep[-1]], axis=1)
#             mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

#         o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
#         mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        
#         out = {'observation': o,
#             'action': a,
#             'reward': r,
#             'next_action': next_a,
#             'next_reward': next_r,
#             'next_state': next_s,
#             'timestep': timestep,
#             'mask': mask}

#         return out


class MCTSLightDarkDataset(Dataset):
    """
    Get a train/test dataset according to the specified settings.
    """
    def __init__(self, config, dataset: List, transform=None):
        self.config = config
        self.dataset = dataset # list of data file name
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        traj = self.dataset[index]
        with open(traj, 'rb') as f:
            traj = pickle.load(f)
        
        if self.transform:
            data = self.transform(traj)
        
        idx = np.random.choice(len(traj) - 2)
        sample = traj[idx]
        
        history = np.asarray(sample[0])
        actions = sample[1]
        p_action = sample[2]
        num_visit_action = sample[3]
        val_node = sample[5]
        if self.config.randomize:
            goal_state = traj[-2]
        total_reward = traj[-1]

        i = np.random.choice(len(actions), p=p_action)
        sampled_next_action = actions[i]

        data = {'action': history[:, 0].tolist(),
                'observation': history[:, 1].tolist(),
                'next_state': history[:, 2].tolist(),
                'reward': history[:, 3].tolist(),
                'next_action': sampled_next_action,
                'total_reward': total_reward}
        
        if self.config.randomize:
            data['goal_state'] = goal_state

        return data


class MCTSBatchMaker():
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dim_action = config.dim_action
        self.dim_observation = config.dim_observation
        self.dim_state = config.dim_state
        self.dim_reward = config.dim_reward
        self.seq_len = config.seq_len


    def __call__(self, data):
        a, o, r, next_a, next_s, goal_s, timestep, mask = [], [], [], [], [], [], [], []
        for d in data:
            # get sequences from dataset
            a.append(np.asarray(d['action']).reshape(1, -1, self.dim_action))
            o.append(np.asarray(d['observation']).reshape(1, -1, self.dim_observation))
            r.append(np.asarray(d['reward']).reshape(1, -1, self.dim_reward))
            next_a.append(np.asarray(d['next_action']).reshape(1, -1, self.dim_action))
            next_s.append(np.asarray(d['next_state']).reshape(1, -1, self.dim_state))
            timestep.append(np.arange(len(d['action'])).reshape(1, -1))
            timestep[-1][timestep[-1] >= self.seq_len] = self.seq_len - 1  # padding cutoff
            if self.config.randomize:
                goal_s.append(np.asarray(d['goal_state']).reshape(1, -1, self.dim_state))

            # padding
            # |FIXME| check padded value & need normalization?
            tlen = o[-1].shape[1]
            o[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen, self.dim_observation)), o[-1]], axis=1)
            a[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen, self.dim_action)), a[-1]], axis=1)
            # a[-1] = np.concatenate([np.ones((1, 31 - tlen, 2)) * -100., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen, self.dim_reward)), r[-1]], axis=1)
            next_s[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen, self.dim_state)), next_s[-1]], axis=1)
            timestep[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen)), timestep[-1]], axis=1)
            if self.config.randomize:
                mask.append(np.concatenate([np.full((1, self.seq_len - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1))
            else:
                mask.append(np.concatenate([np.full((1, self.seq_len - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        if self.config.randomize:
            goal_s = th.from_numpy(np.concatenate(goal_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        
        out = {'observation': o,
            'action': a,
            'reward': r,
            'next_action': next_a,
            'next_state': next_s,
            'timestep': timestep,
            'mask': mask}
        
        if self.config.randomize:
            out['goal_state'] = goal_s

        return out




def get_loader(config, dataset: Dict,
               transform=None, collate_fn=None):
    if config.data_type == 'success':
        dataset = LightDarkDataset(config, dataset, transform)
        if collate_fn == None:
            batcher = BatchMaker(config)
    elif config.data_type == 'mcts':
        dataset = MCTSLightDarkDataset(config, dataset, transform)
        if collate_fn == None:
            batcher = MCTSBatchMaker(config)

    if config.use_sampler:
        sampler = WeightedRandomSampler(dataset.p_sample, config.batch_size)
    else:
        sampler = None

    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=config.shuffle,
                        sampler=sampler,
                        collate_fn=batcher)
    return loader

def get_loader_multi_target(config, dataset):
    dataset = MultiTargetLightDarkDataset(config, dataset)

    loader = DataLoader(dataset, batch_size=1, shuffle=config.shuffle)

    return loader


if __name__ == '__main__':
    @dataclass
    class Settings(Serializable):
        # Dataset
        path: str = 'Learning/dataset'
        data_type: str = 'success' # 'mcts' or 'success'
        train_file: str = 'sim_success_exp_const_30_std0.5_randomize_1' # folder name - mcts / file name - success traj.
        test_file: str = 'sim_success_exp_const_30_std0.5_randomize_1'
        batch_size: int = 2 # 100steps/epoch
        shuffle: bool = True # for using Sampler, it should be False
        use_sampler: bool = False
        max_len: int = 100
        seq_len: int = 31
        randomize: bool = True

        # |TODO| modify to automatically change
        dim_observation: int = 2
        dim_action: int = 2
        dim_state: int = 2
        dim_reward: int = 1

        # Architecture
        model: str = 'CVAE' # GPT or RNN or LSTM or CVAE
        optimizer: str = 'AdamW' # AdamW or AdamWR

        dim_embed: int = 8
        dim_hidden: int = 8

        # for GPT
        dim_head: int = 8
        num_heads: int = 1
        dim_ffn: int = 8 * 4
        num_layers: int = 3

        # for CVAE
        latent_size: int = 32
        dim_condition: int = 32
        # encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
        # decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
        encoder_layer_sizes = [dim_embed, latent_size]
        decoder_layer_sizes = [latent_size, dim_action]

        train_pos_en: bool = False
        use_reward: bool = True
        use_mask_padding: bool = True
        coefficient_loss: float = 1e-3

        dropout: float = 0.1
        action_tanh: bool = False

        # Training
        device: str = 'cuda' if th.cuda.is_available() else 'cpu'
        resume: str = None # checkpoint file name for resuming
        pre_trained: str = None # checkpoint file name for pre-trained model
        # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
        epochs: int = 1000

        # Learning rate
        # |NOTE| using small learning rate, in order to apply warm up
        learning_rate: float = 1e-5
        weight_decay: float = 1e-4
        warmup_step: int = int(1e3)
        # For cosine annealing
        T_0: int = int(1e4)
        T_mult: int = 1
        lr_max: float = 0.01
        lr_mult: float = 0.5

        # Logging
        exp_dir: str = 'Learning/exp'
        model_name: str = 'test'
        print_freq: int = 1000 # per train_steps
        train_eval_freq: int = 1000 # per train_steps
        test_eval_freq: int = 10 # per epochs
        save_freq: int = 100 # per epochs

        log_para: bool = False
        log_grad: bool = False
        eff_grad: bool = False
        print_num_para: bool = True
        print_in_out: bool = False


    config = Settings()
    dataset_path = os.path.join(os.getcwd(), 'Learning/dataset')
    # dataset_filename = 'sim_success_exp_const_30_std0.5_randomize_1'
    dataset_filename = 'test'

    # with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
    #     dataset = pickle.load(f)
    # print('#trajectories of test_dataset:', len(dataset['observation']))

    dataset = glob.glob(f'{dataset_path}/{dataset_filename}/*.pickle')
    print('#trajectories of train_dataset:', len(dataset))

    data_loader = get_loader(config, dataset)
    # data_loader = get_loader_multi_target(config, dataset)

    target = []
    for i in range(500):
        sample = next(iter(data_loader))
        target.append(sample['accumulated_reward'])

    print(target)