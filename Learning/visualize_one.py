import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from dataclasses import dataclass, replace
from simple_parsing import Serializable

from model import GPT2, RNN, LSTM, CVAE
from saver import load_checkpoint
from utils import CosineAnnealingWarmUpRestarts
from load import LightDarkDataset


@dataclass
class Settings(Serializable):
    model: str
    model_name: str
    resume: str = 'best.pth' # checkpoint file name for resuming

    # Architecture
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 128
    dim_hidden: int = 128

    # for GPT
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4
    num_layers: int = 3

    # for CVAE
    latent_size: int = 128
    encoder_layer_sizes = [2, 128]
    decoder_layer_sizes = [128, 2]
    dim_condition: int = 128

    train_pos_en: bool = False
    use_reward: bool = True
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Dataset
    path: str = 'Learning/dataset'
    train_file: str = 'light_dark_long_10K.pickle'
    test_file: str = 'light_dark_long_10K.pickle'
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

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
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
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 10 # per epochs
    save_freq: int = 1000 # per epochs

    # Prediction
    num_pred: int = 100
    current_step: int = 1
    print_in_out: bool = False
    len_input: int = 1
    num_input: int = 1
    num_output: int = 1
    

def collect_data(config, dataset):
    data, target = [], []
    if config.len_input != 1:
        while len(data) < config.num_input:
            index = np.random.choice(len(dataset))
            sample = dataset[index]
            if len(sample['observation']) != config.len_input + 1:
                continue
            
            i = np.random.randint(1, len(sample['observation']))

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

    else:
        while len(data) < config.num_input:
            while True:
                index = np.random.choice(len(dataset))
                sample = dataset[index]
                if len(sample['observation']) == 2:
                    continue
                
                # take first action in sample
                a1 = np.round(sample['action'][1], 4)
                o1 = np.round(sample['observation'][1], 4)
                target_index = []
                for idx in range(len(dataset)):
                    if np.array_equal(np.round(dataset[idx]['action'][1], 4), a1) & np.array_equal(np.round(dataset[idx]['observation'][1], 4), o1):
                        target_index.append(idx)

                if target_index:
                    break

            # Collect multi-targets
            for t in target_index:
                if len(dataset[t]['observation']) == 2:
                    continue
                target.append(dataset[t]['action'][2].reshape(1, -1, 2).squeeze().tolist())

            # truncate & fit interface of sample to model
            o, a, r, next_a, next_s, next_r, timestep, mask = [], [], [], [], [], [], [], []
            i = 2
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

    return data, target

def predict_action(config, model, data):
    model.eval()
    if str(config.device) == 'cuda':
        th.cuda.empty_cache()
    
    with th.no_grad():
        if (config.model == 'RNN') or (config.model == 'GPT'):
            time_start = time.time()
            pred = model(data)
            time_end = time.time()
            pred = pred['action'].tolist()
            inferece_time = time_end - time_start
        elif config.model == 'CVAE':
            time_start = time.time()
            pred = model.inference(data)
            time_end = time.time()
            pred = pred.tolist()
            inferece_time = time_end - time_start

    return pred, inferece_time
    
def main():
    config = Settings(model='GPT', model_name='9.23_dropout0.1_GPT', resume='best.pth')

    dataset_path = os.path.join(os.getcwd(), config.path)
    dataset_filename = config.test_file
    device = config.device
    model_dir = os.path.join(config.exp_dir, config.model_name)

    with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
        dataset = pickle.load(f)
    dataset = LightDarkDataset(config, dataset)
    data, targets = collect_data(config, dataset)

    # with open(os.path.join(dataset_path, 'light_dark_sample_len15.pickle'), 'rb') as f:
    #     sample = pickle.load(f)
    # data, targets = sample['data'], sample['targets']

    if config.model == 'GPT':
        model = GPT2(config).to(device)
    elif config.model == 'RNN':
        model = RNN(config).to(device)
    elif config.model == 'CVAE':
        model = CVAE(config).to(device)
    
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    if config.optimizer == 'AdamW':
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/config.warmup_step, 1))

    elif config.optimizer == 'AdamWR':
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer,
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
        filename = os.path.join(model_dir, config.resume)

        if os.path.isfile(filename):
            start_epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, filename, model, optimizer, scheduler)
            start_epoch += 1
            print("Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch))
        else:
            # |FIXME| using error?exception?logging?
            print("No checkpoint found at '{}'".format(config.resume))
            return


    pred = []
    total_time = 0.

    for d in data:
        for i in range(config.num_output):
            tmp_pred, time = predict_action(config, model, d)

            pred.append(tmp_pred)
            total_time += time

    targets = np.asarray(targets).reshape(-1, 2)
    pred = np.asarray(pred).reshape(-1, 2)

    print(f'Inference time: {total_time / (config.num_input * config.num_output)}')

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.scatter(targets[:,0], targets[:,1], c='red')
    plt.scatter(pred[:,0], pred[:,1], c='blue')
    plt.show()


if __name__ == '__main__':
    main()