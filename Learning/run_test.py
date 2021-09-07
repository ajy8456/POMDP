import os
import time
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
import pickle

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Sampler

# from load import get_loader
from saver import save_checkpoint, load_checkpoint
from utils import CosineAnnealingWarmUpRestarts, log_gradients


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    batch_size: int = 4 # 100steps/epoch
    shuffle: bool = True
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_data: int = 1

    # Architecture
    model: str = 'GPT' # GPT or RNN
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 128
    dim_hidden: int = 128
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4

    num_layers: int = 3

    train_pos_en: bool = False

    dropout: float = 0.0
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
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
    model_name: str = '9.6_log_grad_test_GPT'
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 1000 # per epochs


class TestDataset(Dataset):
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
        self.dim_data = config.dim_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        return data


class BatchMaker():
    def __init__(self, config):
        self.config = config
        self.device = config.device

    def __call__(self, data):
        d, next_d, timestep, mask = [], [], [], []
        for traj in data:
            if len(traj) == 2:
                i = 1
            else:
                i = np.random.randint(1, len(traj) - 1)

            # get sequences from dataset
            d.append(traj[:i].reshape(1, -1, 1))
            next_d.append(traj[i].reshape(1, -1, 1))
            timestep.append(np.arange(0, i).reshape(1, -1))
            timestep[-1][timestep[-1] >= 31] = 31 - 1  # padding cutoff

            # padding
            # |FIXME| check padded value & need normalization?
            tlen = d[-1].shape[1]
            d[-1] = np.concatenate([np.zeros((1, 31 - tlen, 1)), d[-1]], axis=1)
            timestep[-1] = np.concatenate([np.zeros((1, 31 - tlen)), timestep[-1]], axis=1)
            mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

        d = th.from_numpy(np.concatenate(d, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_d = th.from_numpy(np.concatenate(next_d, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        
        out = {'data': d,
            'target': next_d,
            'timestep': timestep,
            'mask': mask}

        return out


class Value(nn.Module):
    def __init__(self, dim_hidden, dim_embed):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class Key(nn.Module):
    def __init__(self, dim_hidden, dim_embed):
        super(Key, self).__init__()
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False)
       
    def forward(self, x):
        x = self.fc1(x)
        return x

class Query(nn.Module):
    def __init__(self, dim_hidden, dim_embed):
        super(Query, self).__init__()
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()

        pe = th.zeros(config.max_len, config.dim_embed)
        position = th.arange(0, config.max_len, dtype=th.float).unsqueeze(1)
        
        div_term = th.exp(th.arange(0, config.dim_embed, 2).float() * (-math.log(10000.0) / config.dim_embed))
        
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1).to(config.device)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x  


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
    
    def forward(self, Q, K, V, attn_mask=None):
        """
        Attention(Q, K, V) = norm(QK)V
        """
        a = th.matmul(Q, K.transpose(-1,-2).float())
        a /= th.sqrt(th.tensor(Q.shape[-1]).float()) # scaled
        
        # Mask(opt.)
        if attn_mask is not None:
            a.masked_fill_(attn_mask, -1e9)

        attn_p = th.softmax(a, -1) # (num_q_seq, num_k_seq)
        attn_v = th.matmul(a, V) # (num_q_seq, dim_hidden)
        return attn_v, attn_p
    

class MultiHeadAttention(th.nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.dim_hidden = config.dim_hidden
        self.dim_head = config.dim_head
        self.num_heads = config.num_heads

        self.W_Q = Query(self.dim_hidden, self.dim_head * self.num_heads)
        self.W_K = Key(self.dim_hidden, self.dim_head * self.num_heads)
        self.W_V = Value(self.dim_hidden, self.dim_head * self.num_heads)
        self.scaled_dot_attn = Attention(config)
        self.fc1 = nn.Linear(self.dim_head * self.num_heads, self.dim_hidden)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]
        # (batch_size, num_heads, num_q_seq, dim_head)
        q_s = self.W_Q(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        # (batch_size, num_heads, num_k_seq, dim_head)
        k_s = self.W_K(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        # (batch_size, num_heads, num_v_seq, dim_head)
        v_s = self.W_V(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)

        # |TODO| check
        # (batch_size, num_heads, num_q_seq, n_k_seq)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).transpose(0,1)

        # |TODO| check shape
        # (batch_size, num_heads, num_q_seq, dim_head), (batch_size, num_heads, num_q_seq, num_k_seq)
        attn_v, attn_p = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (batch_size, num_heads, num_q_seq, num_heads * dim_head)
        attn_v = attn_v.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head)
        # (batch_size, num_q_seq, dim_hidden)
        output = self.fc1(attn_v)
        output = self.dropout(output)

        # (batch_size, num_q_seq, dim_hidden), (batch_size, num_heads, num_q_seq, num_k_seq)
        return output, attn_p
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_hidden = config.dim_hidden
        self.dim_ffn = config.dim_ffn

        self.conv1 = nn.Conv1d(in_channels=self.dim_hidden, out_channels=self.dim_ffn, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.dim_ffn, out_channels=self.dim_hidden, kernel_size=1)
        # |TODO| How to change by config?
        self.act_fn = F.gelu # original: ReLU

    def forward(self, inputs):
        # (batch_size, dim_ffn, num_seq)
        output = self.act_fn(self.conv1(inputs.transpose(1, 2)))
        # (batch_size, num_seq, dim_hidden)
        output = self.conv2(output).transpose(1, 2)

        return output


class GPT2DecoderLayer(nn.Module):
    def __init__(self, config):
        super(GPT2DecoderLayer, self).__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.dim_hidden)
        self.ffn = FeedForwardNetwork(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.dim_hidden)
    
    def forward(self, x, attn_mask):
        # (batch_size, num_dec_seq, dim_hidden), (batch_size, num_heads, num_dec_seq, num_dec_seq)
        self_attn_out, self_attn_prob = self.self_attn(x, attn_mask)
        self_attn_out = self.layer_norm1(x + self_attn_out)

        # (batch_size, num_dec_seq, dim_hidden)
        ffn_out = self.ffn(self_attn_out)
        ffn_outputs = self.layer_norm2(self_attn_out + ffn_out)

        return ffn_outputs, self_attn_prob


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim_data = config.dim_data
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers
        self.action_tanh = config.action_tanh
        self.max_len = config.max_len
        self.seq_len = config.seq_len

        self.embed = nn.Linear(self.dim_data, self.dim_embed)

        # select trainable/fixed positional encoding
        if self.config.train_pos_en:
            self.embed_timestep = nn.Embedding(self.max_len, self.dim_embed)
        else:
            self.pos_embed = PositionalEncoding(self.config)
        
        self.ln = nn.LayerNorm(self.dim_hidden)

        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(GPT2DecoderLayer(self.config))

        self.layers = nn.ModuleList(self.layers)

        self.predict_action = nn.Sequential(*([nn.Linear(self.seq_len * self.dim_hidden, self.dim_data)] + ([nn.Tanh()] if self.action_tanh else [])))

    def forward(self, data):
        batch_size, seq_len = data['data'].shape[0], data['data'].shape[1]

        input_embeddings = self.embed(data['data'])
        
        # select trainable/fixed positional encoding
        if self.config.train_pos_en:
            time_embeddings = self.embed_timestep(data['timestep'])
            input_embeddings = input_embeddings + time_embeddings
        else:
            input_embeddings = self.pos_embed(input_embeddings)

        if 'mask' not in data:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attn_mask = th.ones((batch_size, seq_len), dtype=th.long)
        attn_mask = ~data['mask']

        dec_outputs, attn_prob = self.layers[0](input_embeddings, attn_mask)
        for layer in self.layers[1:]:
            dec_outputs, attn_prob = layer(dec_outputs, attn_mask)

        # get predictions
        pred = self.predict_action(dec_outputs.flatten(start_dim=1))  # predict next action given state
        pred = th.squeeze(pred)

        return pred


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.config = config
        self.dim_data = config.dim_data
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers

        self.embed = nn.Linear(self.dim_data, self.dim_embed)
        
        self.rnn = nn.RNN(input_size=self.dim_embed, hidden_size=self.dim_hidden, num_layers=self.num_layers, batch_first=True)
        self.predict_action = nn.Linear(self.dim_hidden, self.dim_data)


    def forward(self, data):
        batch_size, seq_len = data['data'].shape[0], data['data'].shape[1]
        
        input_embeddings = self.embed(data['data'])

        if 'mask' in data:
            stacked_attention_mask = th.unsqueeze(data['mask'], dim=-1)
            stacked_attention_mask = th.repeat_interleave(~stacked_attention_mask, self.dim_hidden, dim=-1)
            input_embeddings.masked_fill_(stacked_attention_mask, 0)


        h_0 = th.zeros(self.num_layers, batch_size, self.dim_hidden).to(self.config.device)
        output, h_n = self.rnn(input_embeddings, h_0)

        pred = self.predict_action(output[:, -1, :])
        pred = th.squeeze(pred)

        return pred


class Trainer(object):
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """
    def __init__(self,
                 config,
                 loader: th.utils.data.DataLoader,
                 model: th.nn.Module,
                 optimizer: th.optim.Optimizer,
                 loss_fn: Callable[[Dict[str, th.Tensor], Dict[str, th.Tensor]], Dict[str, th.Tensor]],
                 eval_fn: Callable = None,
                 scheduler: th.optim.lr_scheduler = None
                 ):
        """
        Args:
            config: Trainer options.
            model: The model to train.
            optimizer: Optimizer, e.g. `Adam`.
            loss_fn: The function that maps (model, next(iter(loader))) -> cost.
            loader: Iterable data loader.
        """
        self.config = config
        self.loader = loader
        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn

    def train(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        vals = AverageMeter('ME', ':.4e')

        progress = ProgressMeter(len(self.loader),
                                 [batch_time, losses, vals],
                                 prefix="Epoch: [{}]".format(epoch))
        
        self.model.train()

        end = time.time()
        for i, data in enumerate(self.loader):
            target = th.squeeze(data['target'])

            pred = self.model(data)

            loss = self.loss_fn(pred, target)

            # Backprop + Optimize ...
            self.optim.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optim.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # measure elapsed time
            losses.update(loss.item(), data['data'].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config.train_eval_freq == 0:
                self.model.eval()

                with th.no_grad():
                    val = self.eval_fn(pred, target)
                vals.update(val.item(), data['data'].size(0))

                self.model.train()

            if i % self.config.print_freq == 0:
                progress.display(i)
            
        return losses.avg, vals.avg

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Evaluator():
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """
    def __init__(self,
                 config,
                 loader: th.utils.data.DataLoader,
                 model: th.nn.Module,
                 eval_fn: Callable[[Dict[str, th.Tensor], Dict[str, th.Tensor]], Dict[str, th.Tensor]]
                 ):
        self.config = config
        self.loader = loader
        self.model = model
        self.eval_fn = eval_fn
    
    def eval(self, epoch):        
        batch_time = AverageMeter('Time', ':6.3f')
        vals = AverageMeter('MSE', ':.4e')
        progress = ProgressMeter(len(self.loader),
                                 [batch_time, vals],
                                 prefix="Epoch: [{}]".format(epoch))
        
        self.model.eval()
        if str(self.config.device) == 'cuda':
            th.cuda.empty_cache()

        with th.no_grad():
            end = time.time()
            for i, data in enumerate(self.loader):
                target = th.squeeze(data['target'])

                pred = self.model(data)

                val = self.eval_fn(pred, target)

                # measure elapsed time
                vals.update(val.item(), data['data'].size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

        return vals.avg



def main():
    config = Settings()
    # |TODO| go to Setting()
    train_filename = 'data_test.pickle'
    test_filename = 'data_test.pickle'
    dataset_path = os.path.join(os.getcwd(), config.path)
    
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)
    logger = SummaryWriter(model_dir)

    with open(os.path.join(dataset_path, train_filename), 'rb') as f:
        train_dataset = pickle.load(f)
    with open(os.path.join(dataset_path, test_filename), 'rb') as f:
        test_dataset = pickle.load(f)
    
    print('#trajectories of train_dataset:', len(train_dataset))
    print('#trajectories of test_dataset:', len(test_dataset))

    # generate dataloader
    def get_loader(config, dataset: Dict,
               transform=None, collate_fn=None):
        dataset = TestDataset(config, dataset, transform)

        if collate_fn == None:
            batcher = BatchMaker(config)

        loader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=config.shuffle,
                            collate_fn=batcher)
        return loader

    train_loader = get_loader(config, train_dataset)
    test_loader = get_loader(config, test_dataset)

    # model
    device = th.device(config.device)
    if config.model == 'GPT':
        model = GPT2(config).to(device)
    elif config.model == 'RNN':
        model = RNN(config).to(device)
    else:
        # |FIXME| using error?exception?logging?
        print(f'"{config.model}" is not support!! You should select "GPT" or "RNN".')
        return

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

    loss_fn = nn.MSELoss(config)
    eval_fn = nn.MSELoss(config)

    # Trainer & Evaluator
    trainer = Trainer(config=config,
                      loader=train_loader,
                      model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      loss_fn=loss_fn,
                      eval_fn=eval_fn)
    evaluator = Evaluator(config=config,
                          loader=test_loader,
                          model=model,
                          eval_fn=eval_fn)

    start_epoch = 1
    best_error = 10000.

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


    for epoch in range(start_epoch, config.epochs+1):
        print(f'===== Start {epoch} epoch =====')
        
        # Training one epoch
        print("Training...")
        train_loss, train_val = trainer.train(epoch)

        # Logging
        logger.add_scalar('Loss', train_loss, epoch)
        logger.add_scalar('Eval/train', train_val, epoch)
        log_gradients(model, logger, epoch, save_grad=True, save_param=True)

        # evaluating
        if epoch % config.test_eval_freq == 0:
            print("Validating...")
            test_val = evaluator.eval(epoch)

            # save the best model
            if test_val < best_error:
                best_error = test_val

                save_checkpoint('Saving the best model!',
                                os.path.join(model_dir, 'best.pth'),
                                epoch, 
                                best_error, 
                                model, 
                                optimizer, 
                                scheduler
                                )
            
            # Logging
            logger.add_scalar('Eval/test', test_val, epoch)
        
        # save the model
        if epoch % config.save_freq == 0:
            save_checkpoint('Saving...', 
                            os.path.join(model_dir, f'ckpt_epoch_{epoch}.pth'), 
                            epoch, 
                            test_val, 
                            model, 
                            optimizer, 
                            scheduler
                            )

        print(f'===== End {epoch} epoch =====')


if __name__ == '__main__':
    total_time_start = time.time()
    main()
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print("Total Time:", total_time)