import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from dataclasses import dataclass, replace
from simple_parsing import Serializable

from model import GPT2, RNN, LSTM
from saver import load_checkpoint
from utils import CosineAnnealingWarmUpRestarts
from load import LightDarkDataset


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    train_file: str = 'light_dark_long_train_400K.pickle'
    test_file: str = 'light_dark_long_test_100K.pickle'
    batch_size: int = 1 # 100steps/epoch
    shuffle: bool = False # for using Sampler, it should be False
    use_sampler: bool = False
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1

    # Architecture
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 128
    dim_hidden: int = 128
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4

    num_layers: int = 3

    train_pos_en: bool = False
    use_reward: bool = True
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.0
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
    resume: str = 'best.pth' # checkpoint file name for resuming
    # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
    epochs: int = 1500

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
    model_name_GPT: str = '9.14_400Kdata_GPT_drop0.1'
    model_name_RNN: str = '9.10_400Kdata_maskpad_RNN'
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 10 # per epochs
    save_freq: int = 1000 # per epochs

    # Prediction
    num_pred: int = 100
    current_step: int = 1
    print_in_out: bool = False
    num_samples: int = 10
    

def collect_data(config, dataset):
    data, target = [], []
    while len(data) < config.num_samples:
        index = np.random.choice(len(dataset))
        sample = dataset[index]
        if len(sample['observation']) < 15:
            continue
        
        i = np.random.randint(12, len(sample['observation']) - 1)

        # truncate & fit interface of sample to model
        o, a, r, next_a, next_s, next_r, timestep, mask = [], [], [], [], [], [], [], []
        # get sequences from dataset
        o.append(sample['observation'][:i].reshape(1, -1, 2))
        a.append(sample['action'][:i].reshape(1, -1, 2))
        r.append(sample['reward'][:i].reshape(1, -1, 1))
        next_a.append(sample['action'][i].reshape(1, -1, 2))
        next_r.append(sample['reward'][i].reshape(1, -1, 1))
        next_s.append(sample['next_state'][1:i+1].reshape(1, -1, 2))
        timestep.append(np.arange(0, i).reshape(1, -1))
        timestep[-1][timestep[-1] >= 31] = 31 - 1  # padding cutoff
        # padding
        tlen = o[-1].shape[1]
        o[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), o[-1]], axis=1)
        a[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), a[-1]], axis=1)
        # a[-1] = np.concatenate([np.ones((1, 31 - tlen, 2)) * -100., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, 31 - tlen, 1)), r[-1]], axis=1)
        next_s[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), next_s[-1]], axis=1)
        timestep[-1] = np.concatenate([np.zeros((1, 31 - tlen)), timestep[-1]], axis=1)
        mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(config.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(config.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(config.device))
        next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(config.device))
        next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(config.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(config.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(config.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(config.device))
        
        tmp_data = {'observation': o,
            'action': a,
            'reward': r,
            'next_action': next_a,
            'next_reward': next_r,
            'next_state': next_s,
            'timestep': timestep,
            'mask': mask}
        
        data.append(tmp_data)
        target.append(sample['action'][i].reshape(1, -1, 2).squeeze().tolist())

    return data, target

def predict_action(config, model, data):
    model.eval()
    if str(config.device) == 'cuda':
        th.cuda.empty_cache()
    
    with th.no_grad():
        time_start = time.time()
        pred = model(data)
        time_end = time.time()
        pred = pred['action'].tolist()
        inferece_time = time_end - time_start

    return pred, inferece_time
    
def main():
    config = Settings()

    dataset_path = os.path.join(os.getcwd(), config.path)
    dataset_filename = config.test_file
    device = config.device
    model_dir_RNN = os.path.join(config.exp_dir, config.model_name_RNN)
    model_dir_GPT = os.path.join(config.exp_dir, config.model_name_GPT)

    with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
        dataset = pickle.load(f)
    
    dataset = LightDarkDataset(config, dataset)

    data, targets = collect_data(config, dataset)

    model_RNN = RNN(config).to(device)
    model_GPT = GPT2(config).to(device)
    
    optimizer_RNN = th.optim.AdamW(model_RNN.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    optimizer_GPT = th.optim.AdamW(model_GPT.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    if config.optimizer == 'AdamW':
        scheduler_RNN = th.optim.lr_scheduler.LambdaLR(optimizer_RNN, lambda step: min((step+1)/config.warmup_step, 1))
        scheduler_GPT = th.optim.lr_scheduler.LambdaLR(optimizer_GPT, lambda step: min((step+1)/config.warmup_step, 1))
    elif config.optimizer == 'AdamWR':
        scheduler_RNN = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer_RNN,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_max=config.lr_max,
            T_up=config.warmup_step,
            gamma=config.lr_mult
        )
        scheduler_GPT = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer_GPT,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_max=config.lr_max,
            T_up=config.warmup_step,
            gamma=config.lr_mult
        )
    else:
        # |FIXME| using error?exception?logging?
        print(f'"{config.optimizer}" is not support!! You should select "AdamW" or "AdamWR".')
        return


    # load checkpoint for resuming
    if config.resume is not None:
        filename_RNN = os.path.join(model_dir_RNN, config.resume)
        filename_GPT = os.path.join(model_dir_GPT, config.resume)
        if os.path.isfile(filename_RNN):
            start_epoch_RNN, best_error_RNN, model_RNN, optimizer_RNN, scheduler_RNN = load_checkpoint(config, filename_RNN, model_RNN, optimizer_RNN, scheduler_RNN)
            start_epoch_RNN += 1
            print("[RNN]Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch_RNN))
        else:
            # |FIXME| using error?exception?logging?
            print("No checkpoint found at '{}'".format(config.resume))
            return
        if os.path.isfile(filename_GPT):
            start_epoch_GPT, best_error_GPT, model_GPT, optimizer_GPT, scheduler_GPT = load_checkpoint(config, filename_GPT, model_GPT, optimizer_GPT, scheduler_GPT)
            start_epoch_GPT += 1
            print("[GPT]Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch_GPT))
        else:
            # |FIXME| using error?exception?logging?
            print("No checkpoint found at '{}'".format(config.resume))
            return

    pred_RNN = []
    pred_GPT = []
    total_time_GPT = 0.
    total_time_RNN = 0.
    for d in data:
        tmp_pred_RNN, time_RNN = predict_action(config, model_RNN, d)
        tmp_pred_GPT, time_GPT = predict_action(config, model_GPT, d)

        pred_RNN.append(tmp_pred_RNN)
        pred_GPT.append(tmp_pred_GPT)
        total_time_RNN += time_RNN
        total_time_GPT += time_GPT
    
    targets = np.asarray(targets).reshape(2, -1)
    pred_RNN = np.asarray(pred_RNN).reshape(2, -1)
    pred_GPT = np.asarray(pred_GPT).reshape(2, -1)

    print(f'Inference time for RNN: {total_time_RNN / config.num_samples}')
    print(f'Inference time for GPT: {total_time_GPT / config.num_samples}')

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.scatter(targets[0], targets[1], c='red')
    plt.scatter(pred_RNN[0], pred_RNN[1], c='green')
    plt.scatter(pred_GPT[0], pred_GPT[1], c='blue')
    plt.show()


if __name__ == '__main__':
    main()