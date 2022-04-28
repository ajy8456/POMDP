import os
import glob
import time
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import List
import pickle
import torch as th
import numpy as np
from tensorboardX import SummaryWriter

from load import get_loader
from model import GPT2, RNN, LSTM, CVAE, ValueNet
from loss import RegressionLossPolicy, RegressionLossValue, ELBOLoss
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    # data_type: str = 'mcts' # 'mcts' or 'success'
    data_type: str = 'success' # 'mcts' or 'success'
    randomize: bool = True
    filter: float = 51
    data_file: str = 'sim_success_exp_const_30_std0.5_randomize_1' # folder name
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
    model: str = 'ValueNet' # GPT or RNN or LSTM or CVAE or ValueNet or PolicyValueNet
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 16
    dim_hidden: int = 16
    # dim_embed: int = 32
    # dim_hidden: int = 32
    # dim_embed: int = 64
    # dim_hidden: int = 64
    # dim_embed: int = 128
    # dim_hidden: int = 128

    # for GPT
    dim_head: int = 16
    num_heads: int = 1
    dim_ffn: int = 16 * 4
    num_layers: int = 3
    # dim_head: int = 32
    # num_heads: int = 1
    # dim_ffn: int = 32 * 4
    # num_layers: int = 3
    # dim_head: int = 64
    # num_heads: int = 1
    # dim_ffn: int = 64 * 4
    # num_layers: int = 3
    # dim_head: int = 128
    # num_heads: int = 1
    # dim_ffn: int = 128 * 3
    # num_layers: int = 3

    # for CVAE
    latent_size: int = 16
    dim_condition: int = 16
    # latent_size: int = 32
    # dim_condition: int = 32
    # latent_size: int = 64
    # dim_condition: int = 64
    # latent_size: int = 128
    # dim_condition: int = 128
    encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
    decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
    # encoder_layer_sizes = [dim_embed, latent_size]
    # decoder_layer_sizes = [latent_size, dim_action]

    train_pos_en: bool = False
    use_reward: bool = True
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    resume: str = 'best.pth' # checkpoint file name for resuming
    pre_trained: str = None

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
    model_name: str = '4.10_ValueNet'
    # model_name: str = '4.10_ValueNet_clip'
    # model_name: str = '4.11_ValueNet_x2'

    num_data: int = 100
    

def main():
    config = Settings()
    # |TODO| go to Setting()
    data_filename = config.data_file
    dataset_path = os.path.join(os.getcwd(), config.path)
    
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)

    if config.data_type == 'success':
        # with open(os.path.join(dataset_path, train_filename), 'rb') as f:
        #     train_dataset = pickle.load(f)
        # with open(os.path.join(dataset_path, test_filename), 'rb') as f:
        #     test_dataset = pickle.load(f)

        dataset = glob.glob(f'{dataset_path}/{data_filename}/*.pickle')
        # test_dataset = glob.glob(f'{dataset_path}/{test_filename}/*.pickle')
        test_dataset = dataset[-200000:]

        print('#trajectories of test_dataset:', len(test_dataset))
    
    elif config.data_type == 'mcts':
        dataset = glob.glob(f'{dataset_path}/{data_filename}/*.pickle')
        test_dataset = dataset[-30000:]

        if config.filter:
            filtered_data_test = []
            total_reward_filt = []
            total_reward_not_filt = []
            avg_total_reward_not_filt = 0
            avg_total_reward_filt = 0

            for data in test_dataset:
                with open(data, 'rb') as f:
                    traj = pickle.load(f)
                    if traj[-1] > config.filter:
                        filtered_data_test.append(data)
                        
            total_reward_not_filt_std = np.std(np.asarray(total_reward_not_filt))
            total_reward_filt_std = np.std(np.asarray(total_reward_filt))
            print('Average of total reward(not filtered):', avg_total_reward_not_filt/len(test_dataset))
            print('std of total reward(not filtered):', total_reward_not_filt_std)
            print('Average of total reward(filtered):', avg_total_reward_filt/len(filtered_data_test))
            print('std of total reward(filtered):', total_reward_filt_std)
            
            test_dataset = filtered_data_test
    
        print('#trajectories of test_dataset:', len(test_dataset))

    # generate dataloader
    data_loader = get_loader(config, test_dataset)

    # model
    device = th.device(config.device)
    if config.model == 'GPT':
        model = GPT2(config).to(device)
    elif config.model == 'RNN':
        model = RNN(config).to(device)
    elif config.model == 'LSTM':
        model = LSTM(config).to(device)
    elif config.model == 'CVAE' or config.model == 'PolicyValueNet':
        model = CVAE(config).to(device)
    elif config.model == 'ValueNet':
        model = ValueNet(config).to(device)
    else:
        raise Exception(f'"{config.model}" is not support!! You should select "GPT", "RNN", "LSTM", "CVAE", "ValueNet", or "PolicyValueNet.')

    # optimizer
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    # learning rate scheduler
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
        raise Exception(f'"{config.optimizer}" is not support!! You should select "AdamW" or "AdamWR".')

    # load checkpoint for resuming
    if config.resume is not None:
        filename = os.path.join(model_dir, config.resume)
        if os.path.isfile(filename):
            start_epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, filename, model, optimizer, scheduler)
            start_epoch += 1
            print("Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch))
        else:
            raise Exception("No checkpoint found at '{}'".format(config.resume))


    targets = []
    preds = []

    for n in range(config.num_data):
        data = next(iter(data_loader))

        # predict next action

        with th.no_grad():
            time_start = time.time()
            pred = model(data)
            time_end = time.time()
            infer_time = time_end - time_start
        
        targets.append(data['accumulated_reward'].item())
        preds.append(pred.item())

    print(targets, preds)


if __name__ == '__main__':
    main()