from itertools import accumulate
import os
import time
import random
import numpy as np
import torch as th
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from torch.utils import data

from Learning.model import GPT2, RNN, LSTM, CVAE, ValueNet
from Learning.saver import load_checkpoint
from Learning.utils import CosineAnnealingWarmUpRestarts


@dataclass
class SettingValue(Serializable):
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1
    max_len: int = 100
    seq_len: int = 31

    # Architecture
    model: str = 'ValueNet'
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 16
    dim_hidden: int = 16
    # dim_embed: int = 128
    # dim_hidden: int = 128

    # for GPT
    dim_head: int = 16
    num_heads: int = 1
    dim_ffn: int = 16 * 4
    num_layers: int = 3
    # dim_head: int = 128
    # num_heads: int = 1
    # dim_ffn: int = 128 * 3
    # num_layers: int = 3
    
    # for CVAE
    latent_size: int = 16
    dim_condition: int = 16
    # latent_size: int = 128
    # dim_condition: int = 128
    encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
    decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
    # encoder_layer_sizes = [dim_embed, latent_size]
    # decoder_layer_sizes = [latent_size, dim_action]

    train_pos_en: bool = False
    use_reward: bool = True
    use_mask_padding: bool = True
    randomize: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
    resume: str = 'best.pth' # checkpoint file name for resuming
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
    # model_name: str = '4.17_ValueNet'
    model_name: str = '4.25_ValueNet_sim_success_2'

    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 10 # per epochs
    save_freq: int = 1000 # per epochs

    # Prediction
    print_in_out: bool = False
    variance: float = 0.5


class GuideValueNet():
    def __init__(self, config, const):
        self.config = config
        self.const = const

        # model
        self.model_dir = os.path.join(config.exp_dir, config.model_name)
        self.device = th.device(config.device)
        self.model = ValueNet(config).to(self.device)

        self.model.eval()

        # optimizer
        self.optimizer = th.optim.AdamW(self.model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
        
        # learning rate scheduler
        if config.optimizer == 'AdamW':
            self.scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min((step+1)/config.warmup_step, 1))
        elif config.optimizer == 'AdamWR':
            self.scheduler = CosineAnnealingWarmUpRestarts(
                optimizer=self.optimizer,
                T_0=config.T_0,
                T_mult=config.T_mult,
                eta_max=config.lr_max,
                T_up=config.warmup_step,
                gamma=config.lr_mult
            )
        else:
            raise Exception(f'"{config.optimizer}" is not support!! You should select "AdamW" or "AdamWR".')

        # load checkpoint for resuming
        if config.resume is not None:
            filename = os.path.join(self.model_dir, config.resume)
            if os.path.isfile(filename):
                start_epoch, _, self.model, self.optimizer, self.scheduler = load_checkpoint(config, filename, self.model, self.optimizer, self.scheduler)
                start_epoch += 1
                print("Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch))
            else:
                raise Exception("No checkpoint found at '{}'".format(config.resume))

    def sample(self, history, goal):
        """
        infer accumulated reward using neural network
        args:
            history: tuple of current history ((a0, o0, s'0, r0)), (a1, o1, s'1, r1), ... )
        returns:
            pred(float): prediction of accumulated reward
            infer_time(float)
        """
        # fitting form of traj to input of network
        data = self._traj2data(history, goal)

        # predict next action

        with th.no_grad():
            time_start = time.time()
            pred = self.model(data).item()
            time_end = time.time()
            infer_time = time_end - time_start

        return pred, infer_time
    
    def _traj2data(self, history, goal):
        """
        interface matching for neural network 
        """
        o, a, r, goal_s, timestep, mask = [], [], [], [], [], []
        i = 2

        # get sequences from dataset
        for i, h in enumerate(history):
            a.append(h[0])
            o.append(h[1])
            r.append(h[3])
            timestep.append(i)
        goal_s.append(goal)
        o = np.asarray(o).reshape(1, -1, 2)
        a = np.asarray(a).reshape(1, -1, 2)
        r = np.asarray(r).reshape(1, -1, 1)
        goal_s = np.asarray(goal_s).reshape(1, -1, 2)
        timestep = np.asarray(timestep).reshape(1, -1)

        # padding
        tlen = timestep.shape[1]
        if self.config.randomize:
            mask.append(np.concatenate([np.full((1, self.config.seq_len - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1))
        else:
            mask.append(np.concatenate([np.full((1, self.config.seq_len - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen, 2)), o], axis=1)).to(dtype=th.float32, device=th.device(self.config.device))
        a = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen, 2)), a], axis=1)).to(dtype=th.float32, device=th.device(self.config.device))
        r = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen, 1)), r], axis=1)).to(dtype=th.float32, device=th.device(self.config.device))
        timestep = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen)), timestep], axis=1)).to(dtype=th.long, device=th.device(self.config.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        goal_s = th.from_numpy(goal_s).to(dtype=th.float32, device=th.device(self.device))

        data = {'observation': o,
            'action': a,
            'reward': r,
            'goal_state': goal_s,
            'timestep': timestep,
            'mask': mask}

        return data


if __name__ == '__main__':
    nn_config = SettingValue()
    guide_poilcy = GuideValueNet(nn_config, [0.5, 0.5])

    history = (((0, 0), (5.296355492413373, -0.24048282626180972), (2.0623819119018396, 2.1958711763514596), 0), ((2.796929254158842, 2.0435689830185737), (4.690556164130332, 4.491229440722575), (4.747127206199204, 4.527105117508303), -1), )
    goal = (0, 0)
    accumulated_reward, inference_time = guide_poilcy.sample(history, goal)
    print(accumulated_reward, type(accumulated_reward))
    print(inference_time)