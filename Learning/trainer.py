import time
import torch as th
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)


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
                 loss_fn: Callable[[th.Tensor, th.Tensor], th.Tensor],
                 eval_fn: Callable = None
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
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn

    def train(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        vals = AverageMeter('MAE', ':.4e')
        progress = ProgressMeter(len(self.loader),
                                 [batch_time, losses, vals],
                                 prefix="Epoch: [{}]".format(epoch))
        
        self.model.train()

        end = time.time()
        for i, data in enumerate(self.loader):
            observations, actions, timestep, attn_mask = data['observation'], data['action'], data['timesteps'], data['mask']
            target_actions = data['next_action']

            pred_actions = self.model(observations, actions, attn_mask)
            # pred_actions = self.model(observations, actions, timestep, attn_mask)

            loss = self.loss_fn(pred_actions, target_actions)

            # Backprop + Optimize ...
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # measure elapsed time
            losses.update(loss.item(), data['observation'].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config.train_eval_freq == 0:
                self.model.eval()
                with th.no_grad():
                    val = self.eval_fn(pred_actions, target_actions)
                vals.update(val.item(), data['observation'].size(0))
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
