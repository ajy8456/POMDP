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
        losses_total = AverageMeter('Total Loss', ':.4e')
        losses_action = AverageMeter('Action Loss', ':.4e')
        losses_reward = AverageMeter('Reward Loss', ':.4e')
        vals_action = AverageMeter('Action MAE', ':.4e')
        vals_reward = AverageMeter('Reward MSE', ':.4e')

        progress = ProgressMeter(len(self.loader),
                                 [batch_time, losses_total, vals_action],
                                 prefix="Epoch: [{}]".format(epoch))
        
        self.model.train()

        end = time.time()
        for i, data in enumerate(self.loader):
            target = {}
            target_action = th.squeeze(data['next_action'])
            target['action'] = target_action
            # if self.config.use_reward:
            #     target_reward = th.squeeze(data['next_reward'])
            #     target['reward'] = target_reward

            pred = self.model(data)

            loss = self.loss_fn(pred, target)

            # Backprop + Optimize ...
            self.optim.zero_grad()
            loss['total'].backward()
            th.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optim.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # measure elapsed time
            losses_total.update(loss['total'].item(), data['observation'].size(0))
            losses_action.update(loss['action'].item(), data['observation'].size(0))
            # if self.config.use_reward:
            #     losses_reward.update(loss['reward'].item(), data['observation'].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config.train_eval_freq == 0:
                self.model.eval()

                with th.no_grad():
                    val = self.eval_fn(pred, target)
                vals_action.update(val['action'].item(), data['observation'].size(0))
                # if self.config.use_reward:
                #     vals_reward.update(val['reward'].item(), data['observation'].size(0))

                self.model.train()

            if i % self.config.print_freq == 0:
                progress.display(i)
            
            losses = {}
            losses['total'] = losses_total.avg
            losses['action'] = losses_action.avg
            # if self.config.use_reward:
            #     losses['reward'] = losses_reward.avg

            vals = {}
            vals['action'] = vals_action.avg
            # if self.config.use_reward:
            #     vals['reward'] = vals_reward.avg

        return losses, vals

    
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
