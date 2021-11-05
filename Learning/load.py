import os
import pickle
import numpy as np
import torch as th
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from dataclasses import dataclass, replace
from simple_parsing import Serializable


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

        # # for WeightedRandomSampler
        # self.p_sample = dataset['p_sample']

    def __len__(self):
        return len(self.dataset['observation'])

    def __getitem__(self, index):
        observation = self.dataset['observation'][index]
        action = self.dataset['action'][index]
        reward = self.dataset['reward'][index]
        next_state = self.dataset['next_state'][index]
        traj_len = self.dataset['traj_len'][index]
        sample = {'observation': observation,
                  'action': action,
                  'reward': reward,
                  'next_state': next_state,
                  'traj_len': traj_len}
        
        if self.transform:
            sample = self.transform(sample)

        return sample


class BatchMaker():
    def __init__(self, config):
        self.config = config
        self.device = config.device

    def __call__(self, data):
        o, a, r, next_a, next_s, next_r, timestep, mask = [], [], [], [], [], [], [], []
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
            next_r.append(traj['reward'][i].reshape(1, -1, 1))
            next_s.append(traj['next_state'][1:i+1].reshape(1, -1, 2))
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
            mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        
        out = {'observation': o,
            'action': a,
            'reward': r,
            'next_action': next_a,
            'next_reward': next_r,
            'next_state': next_s,
            'timestep': timestep,
            'mask': mask}

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
        traj = {'observation': observation,
                  'action': action,
                  'reward': reward,
                  'next_state': next_state,
                  'traj_len': traj_len}

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
        o, a, r, next_a, next_s, next_r, timestep, mask = [], [], [], [], [], [], [], []
        o.append(traj['observation'][:i].reshape(-1, 2))
        a.append(traj['action'][:i].reshape(-1, 2))
        r.append(traj['reward'][:i].reshape(-1, 1))
        next_a.append(np.round(traj['action'][i], 4).reshape(-1, 2))
        next_r.append(traj['reward'][i].reshape(-1, 1))
        next_s.append(traj['next_state'][1:i+1].reshape(-1, 2))
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
        mask.append(np.concatenate([np.full(31 - tlen, False, dtype=bool), np.full(tlen, True, dtype=bool)]))

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        # next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
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
        observation = self.dataset['observation'][index]
        action = self.dataset['action'][index]
        reward = self.dataset['reward'][index]
        next_state = self.dataset['next_state'][index]
        traj_len = self.dataset['traj_len'][index]
        sample = {'observation': observation,
                  'action': action,
                  'reward': reward,
                  'next_state': next_state,
                  'traj_len': traj_len}
        
        if self.transform:
            sample = self.transform(sample)

        return sample    



def get_loader(config, dataset: Dict,
               transform=None, collate_fn=None):
    dataset = LightDarkDataset(config, dataset, transform)

    if config.use_sampler:
        sampler = WeightedRandomSampler(dataset.p_sample, config.batch_size)
    else:
        sampler = None

    if collate_fn == None:
        batcher = BatchMaker(config)

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
        train_file: str = 'light_dark_long_train_400K.pickle'
        test_file: str = 'light_dark_long_test_100K.pickle'
        batch_size: int = 1 # 100steps/epoch
        shuffle: bool = True # for using Sampler, it should be False
        use_sampler: bool = False
        max_len: int = 100
        seq_len: int = 31
        # |TODO| modify to automatically change
        dim_observation: int = 2
        dim_action: int = 2
        dim_state: int = 2
        dim_reward: int = 1

        # Architecture
        model: str = 'GPT' # GPT or RNN or LSTM or CVAE
        optimizer: str = 'AdamW' # AdamW or AdamWR

        dim_embed: int = 16
        dim_hidden: int = 16

        # for GPT
        dim_head: int = 16
        num_heads: int = 1
        dim_ffn: int = 16 * 4
        num_layers: int = 4

        # for CVAE
        latent_size: int = 128
        encoder_layer_sizes = [2, 32, 16]
        decoder_layer_sizes = [32, 16, 2]
        dim_condition: int = 128

        train_pos_en: bool = False
        use_reward: bool = True
        use_mask_padding: bool = True
        coefficient_loss: float = 1e-3

        dropout: float = 0.1
        action_tanh: bool = False

        # Training
        device: str = 'cuda' if th.cuda.is_available() else 'cpu'
        resume: str = None # checkpoint file name for resuming
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
        model_name: str = '10.4_GPT_dim16_layer4'
        print_freq: int = 1000 # per train_steps
        train_eval_freq: int = 1000 # per train_steps
        test_eval_freq: int = 10 # per epochs
        save_freq: int = 100 # per epochs

        log_para: bool = True
        log_grad: bool = True
        eff_grad: bool = False
        print_num_para: bool = False
        print_in_out: bool = False


    config = Settings()
    dataset_path = os.path.join(os.getcwd(), 'Learning/dataset')
    dataset_filename = 'light_dark_long_mini.pickle'

    with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
        dataset = pickle.load(f)
    print('#trajectories of test_dataset:', len(dataset['observation']))

    data_loader = get_loader_multi_target(config, dataset)

    sample = next(iter(data_loader))

    print(sample)