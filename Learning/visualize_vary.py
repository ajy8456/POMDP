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
    # model_name_RNN: str = '9.23_dropout0.1_RNN'
    # model_name_GPT: str = '9.24_dropout0.1_GPT_reduce_dim_ffn'
    # model_name_CVAE: str = '9.27_CVAE'
    resume: str = 'best.pth' # checkpoint file name for resuming

    # Architecture
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 8
    dim_hidden: int = 8

    # for GPT
    dim_head: int = 8
    num_heads: int = 1
    dim_ffn: int = 8 * 4
    num_layers: int = 6

    # for CVAE
    latent_size: int = 128
    encoder_layer_sizes = [2, 32, 8]
    decoder_layer_sizes = [32, 8, 2]
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
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 10 # per epochs
    save_freq: int = 1000 # per epochs

    # Prediction
    num_pred: int = 100
    current_step: int = 1
    print_in_out: bool = False
    len_input: int = 15
    num_input: int = 1
    num_output: int = 10
    

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
    # config_RNN = Settings(model='RNN', model_name='9.23_dropout0.1_RNN', resume='best.pth')
    config_GPT = Settings(model='GPT', model_name='10.3_GPT_dim8_layer6', resume='best.pth')
    config_CVAE = Settings(model='CVAE', model_name='10.3_CVAE_dim8_layer6', resume='best.pth')

    dataset_path = os.path.join(os.getcwd(), config_GPT.path)
    dataset_filename = config_GPT.test_file
    device = config_GPT.device
    # model_dir_RNN = os.path.join(config_RNN.exp_dir, config_RNN.model_name)
    model_dir_GPT = os.path.join(config_GPT.exp_dir, config_GPT.model_name)
    model_dir_CVAE = os.path.join(config_CVAE.exp_dir, config_CVAE.model_name)

    with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
        dataset = pickle.load(f)
    dataset = LightDarkDataset(config_GPT, dataset)
    data, targets = collect_data(config_GPT, dataset)

    # with open(os.path.join(dataset_path, 'light_dark_sample_len15.pickle'), 'rb') as f:
    #     sample = pickle.load(f)
    # data, targets = sample['data'], sample['targets']

    # model_RNN = RNN(config_RNN).to(device)
    model_GPT = GPT2(config_GPT).to(device)
    model_CVAE = CVAE(config_CVAE).to(device)
    
    # optimizer_RNN = th.optim.AdamW(model_RNN.parameters(),
    #                            lr=config_RNN.learning_rate,
    #                            weight_decay=config_RNN.weight_decay)
    optimizer_GPT = th.optim.AdamW(model_GPT.parameters(),
                               lr=config_GPT.learning_rate,
                               weight_decay=config_GPT.weight_decay)
    optimizer_CVAE = th.optim.AdamW(model_CVAE.parameters(),
                               lr=config_CVAE.learning_rate,
                               weight_decay=config_CVAE.weight_decay)
    
    if config_GPT.optimizer == 'AdamW':
        # scheduler_RNN = th.optim.lr_scheduler.LambdaLR(optimizer_RNN, lambda step: min((step+1)/config_RNN.warmup_step, 1))
        scheduler_GPT = th.optim.lr_scheduler.LambdaLR(optimizer_GPT, lambda step: min((step+1)/config_GPT.warmup_step, 1))
        scheduler_CVAE = th.optim.lr_scheduler.LambdaLR(optimizer_CVAE, lambda step: min((step+1)/config_CVAE.warmup_step, 1))
    elif config_GPT.optimizer == 'AdamWR':
        # scheduler_RNN = CosineAnnealingWarmUpRestarts(
        #     optimizer=optimizer_RNN,
        #     T_0=config_RNN.T_0,
        #     T_mult=config_RNN.T_mult,
        #     eta_max=config_RNN.lr_max,
        #     T_up=config_RNN.warmup_step,
        #     gamma=config_RNN.lr_mult
        # )
        scheduler_GPT = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer_GPT,
            T_0=config_GPT.T_0,
            T_mult=config_GPT.T_mult,
            eta_max=config_GPT.lr_max,
            T_up=config_GPT.warmup_step,
            gamma=config_GPT.lr_mult
        )
        scheduler_CVAE = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer_CVAE,
            T_0=config_CVAE.T_0,
            T_mult=config_CVAE.T_mult,
            eta_max=config_CVAE.lr_max,
            T_up=config_CVAE.warmup_step,
            gamma=config_CVAE.lr_mult
        )
    else:
        # |FIXME| using error?exception?logging?
        print(f'"{config_GPT.optimizer}" is not support!! You should select "AdamW" or "AdamWR".')
        return


    # load checkpoint for resuming
    if config_GPT.resume is not None:
        # filename_RNN = os.path.join(model_dir_RNN, config_RNN.resume)
        filename_GPT = os.path.join(model_dir_GPT, config_GPT.resume)
        filename_CVAE = os.path.join(model_dir_CVAE, config_CVAE.resume)

        # if os.path.isfile(filename_RNN):
        #     start_epoch_RNN, best_error_RNN, model_RNN, optimizer_RNN, scheduler_RNN = load_checkpoint(config_RNN, filename_RNN, model_RNN, optimizer_RNN, scheduler_RNN)
        #     start_epoch_RNN += 1
        #     print("[RNN]Loaded checkpoint '{}' (epoch {})".format(config_RNN.resume, start_epoch_RNN))
        # else:
        #     # |FIXME| using error?exception?logging?
        #     print("No checkpoint found at '{}'".format(config_RNN.resume))
        #     return

        if os.path.isfile(filename_GPT):
            start_epoch_GPT, best_error_GPT, model_GPT, optimizer_GPT, scheduler_GPT = load_checkpoint(config_GPT, filename_GPT, model_GPT, optimizer_GPT, scheduler_GPT)
            start_epoch_GPT += 1
            print("[GPT]Loaded checkpoint '{}' (epoch {})".format(config_GPT.resume, start_epoch_GPT))
        else:
            # |FIXME| using error?exception?logging?
            print("No checkpoint found at '{}'".format(config_GPT.resume))
            return

        if os.path.isfile(filename_CVAE):
            start_epoch_CVAE, best_error_CVAE, model_CVAE, optimizer_CVAE, scheduler_CVAE = load_checkpoint(config_CVAE, filename_CVAE, model_CVAE, optimizer_CVAE, scheduler_CVAE)
            start_epoch_CVAE += 1
            print("[CVAE]Loaded checkpoint '{}' (epoch {})".format(config_CVAE.resume, start_epoch_CVAE))
        else:
            # |FIXME| using error?exception?logging?
            print("No checkpoint found at '{}'".format(config_CVAE.resume))
            return

    # pred_RNN = []
    pred_GPT = []
    pred_CVAE = []
    # total_time_RNN = 0.
    total_time_GPT = 0.
    total_time_CVAE = 0.
    for d in data:
        for i in range(config_GPT.num_output):
            # tmp_pred_RNN, time_RNN = predict_action(config_RNN, model_RNN, d)
            tmp_pred_GPT, time_GPT = predict_action(config_GPT, model_GPT, d)
            tmp_pred_CVAE, time_CVAE = predict_action(config_CVAE, model_CVAE, d)

            # pred_RNN.append(tmp_pred_RNN)
            pred_GPT.append(tmp_pred_GPT)
            pred_CVAE.append(tmp_pred_CVAE)
            # total_time_RNN += time_RNN
            total_time_GPT += time_GPT
            total_time_CVAE += time_CVAE
    
    targets = np.asarray(targets).reshape(-1, 2)
    # pred_RNN = np.asarray(pred_RNN).reshape(-1, 2)
    pred_GPT = np.asarray(pred_GPT).reshape(-1, 2)
    pred_CVAE = np.asarray(pred_CVAE).reshape(-1, 2)

    # print(f'Inference time for RNN: {total_time_RNN / (config_RNN.num_input * config_RNN.num_output)}')
    print(f'Inference time for GPT: {total_time_GPT / (config_GPT.num_input * config_GPT.num_output)}')
    print(f'Inference time for CVAE: {total_time_CVAE / (config_CVAE.num_input * config_CVAE.num_output)}')

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.scatter(targets[:,0], targets[:,1], c='red')
    # plt.scatter(pred_RNN[:,0], pred_RNN[:,1], c='green')
    plt.scatter(pred_GPT[:,0], pred_GPT[:,1], c='blue')
    plt.scatter(pred_CVAE[:,0], pred_CVAE[:,1], c='black')
    plt.show()


if __name__ == '__main__':
    main()