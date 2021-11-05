import time
import torch as th
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)

from trainer import AverageMeter, ProgressMeter

import os
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import List
import pickle
import torch as th
from tensorboardX import SummaryWriter

from load import get_loader, get_loader_multi_target
from model import GPT2, RNN, LSTM, CVAE
from loss import RegressionLoss, ELBOLoss
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    dataset_file: str = 'light_dark_long_10K.pickle'
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
    model: str = 'CVAE' # GPT or RNN or LSTM or CVAE
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 16
    dim_hidden: int = 16

    # for GPT
    dim_head: int = 16
    num_heads: int = 1
    dim_ffn: int = 16 * 4
    num_layers: int = 3

    # for CVAE
    latent_size: int = 16
    dim_condition: int = 16
    encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
    decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]

    train_pos_en: bool = False
    use_reward: bool = True
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    resume = ['ckpt_epoch_100.pth', 'ckpt_epoch_200.pth', 'ckpt_epoch_300.pth', 'ckpt_epoch_400.pth', 'ckpt_epoch_500.pth', 'ckpt_epoch_600.pth', 'ckpt_epoch_700.pth', 'ckpt_epoch_800.pth', 'ckpt_epoch_900.pth', 'ckpt_epoch_1000.pth'] # checkpoint file name for resuming
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
    model_name: str = '10.12_CVAE_dim16_recur1_scratch'
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 10 # per epochs
    save_freq: int = 100 # per epochs

    log_para: bool = True
    log_grad: bool = True
    eff_grad: bool = False
    print_num_para: bool = False
    print_in_out: bool = False

    # prediction
    num_pred: int = 10 # for CVAE, NNMSE


class MultiTargetEvaluator():
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """
    def __init__(self,
                 config,
                 loader: th.utils.data.DataLoader,
                 model: th.nn.Module,
                 eval_fn: Callable):
        self.config = config
        self.loader = loader
        self.model = model
        self.eval_fn = eval_fn
    
    def eval(self, epoch):        
        batch_time = AverageMeter('Time', ':6.3f')
        nmse = AverageMeter('Nearest MSE', ':4e')

        progress = ProgressMeter(len(self.loader),
                                    [batch_time, nmse],
                                    prefix="Epoch: [{}]".format(epoch))

        self.model.eval()
        if str(self.config.device) == 'cuda':
            th.cuda.empty_cache()

        with th.no_grad():
            end = time.time()
            for i, data in enumerate(self.loader):
                target = data['next_action'].reshape(-1, 2)

                if self.config.model == 'CVAE':
                    pred_sample = []

                    # generate multiple prediction for NNMSE
                    for _ in range(self.config.num_pred):
                        pred_sample.append(self.model.inference(data))

                    pred = {'action': th.stack(pred_sample).reshape(-1, self.config.dim_action)}

                else:
                    pred = self.model(data)
                    
                val = self.eval_fn(pred, target)

                nmse.update(val.item(), data['observation'].size(0))
                # if self.config.use_reward:
                #     vals_reward.update(val['reward'].item(), data['observation'].size(0))
                
                test_val = nmse.avg
                # if self.config.use_reward:
                #     vals['reward'] = vals_reward.avg
            
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

        return test_val


def NMSE(pred: Dict, target: th.Tensor): # Nearest MSE (uni-prediction to multi-target)
    val = th.norm((pred['action'] - target), 2, dim=-1)
    val = th.min(val)
    return val

# |NOTE| change to faster
def NNMSE(pred: Dict, target: th.Tensor): # Near-Nearest MSE (multi-prediction to multi-target)
    mse = th.empty(pred['action'].size(0), target.size(0)).fill_(1e9)
    for m in range(pred['action'].size(0)):
        for n in range(target.size(0)):
            mse[m][n] = th.norm(pred['action'][m] - target[n], 2)
    val = th.min(mse)
    return val


def main():
    config = Settings()
    dataset_filename = config.dataset_file
    dataset_path = os.path.join(os.getcwd(), config.path)
    
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)

    logger = SummaryWriter(model_dir)

    with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
        dataset = pickle.load(f)
    
    print('#trajectories of dataset:', len(dataset['observation']))

    # generate dataloader
    data_loader = get_loader_multi_target(config, dataset)

    # model
    device = th.device(config.device)
    if config.model == 'GPT':
        model = GPT2(config).to(device)
    elif config.model == 'RNN':
        model = RNN(config).to(device)
    elif config.model == 'LSTM':
        model = LSTM(config).to(device)
    elif config.model == 'CVAE':
        model = CVAE(config).to(device)
    else:
        raise Exception(f'"{config.model}" is not support!! You should select "GPT", "RNN", or "LSTM".')

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

    # Metric
    if config.model == 'CVAE':
        eval_fn = NNMSE
    else:
        eval_fn = NMSE

    # Trainer & Evaluator

    evaluator = MultiTargetEvaluator(config=config,
                          loader=data_loader,
                          model=model,
                          eval_fn=eval_fn)

    # load checkpoint for resuming
    for ckpt in config.resume:
        filename = os.path.join(model_dir, ckpt)
        if os.path.isfile(filename):
            epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, filename, model, optimizer, scheduler)
            print("Loaded checkpoint '{}' (epoch {})".format(ckpt, epoch))
        else:
            raise Exception("No checkpoint found at '{}'".format(ckpt))

        print(f'===== Evaluate {epoch} epoch =====')
        
        test_val = evaluator.eval(epoch)
            
        # Logging
        logger.add_scalar('Eval/Near-Nearest MSE', test_val, epoch)
        
        print(f'===== End {epoch} epoch =====')


if __name__ == '__main__':
    main()