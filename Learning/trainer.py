from itertools import accumulate
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
        if self.config.model == 'CVAE':
            losses_elbo = AverageMeter('ELBO', ':4e')
            losses_recon = AverageMeter('Reconstruction Error', ':4e')
            losses_kld = AverageMeter('KL-divergence', ':.4e')
            vals_elbo = AverageMeter('ELBO', ':4e')
            vals_recon = AverageMeter('Reconstruction Error', ':4e')
            vals_kld = AverageMeter('KL-divergence', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, losses_elbo, vals_elbo],
                                     prefix="Epoch: [{}]".format(epoch))
        
        elif self.config.model == 'ValueNet':
            losses_total = AverageMeter('MSELoss', ':.4e')
            vals_total = AverageMeter('MSELoss', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, losses_total, vals_total],
                                     prefix="Epoch: [{}]".format(epoch))

        elif self.config.model == 'PolicyValueNet':
            losses_total = AverageMeter('Total Loss', ':.4e')
            losses_elbo = AverageMeter('Action ELBOLoss', ':.4e')
            losses_recon = AverageMeter('Action Reconstruction Error', ':.4e')
            losses_kld = AverageMeter('Action KL-divergence', ':.4e')
            losses_mse = AverageMeter('Accumulated Reward MSELoss', ':.4e')

            vals_total = AverageMeter('Total Eval', ':.4e')
            vals_elbo = AverageMeter('Action ELBOLoss', ':.4e')
            vals_recon = AverageMeter('Action Reconstruction Error', ':.4e')
            vals_kld = AverageMeter('Action KL-divergence', ':.4e')
            vals_mse = AverageMeter('Accumulated Reward MSELoss', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, losses_total, vals_total],
                                     prefix="Epoch: [{}]".format(epoch))
                        
        else:
            losses_total = AverageMeter('Total Loss', ':.4e')
            losses_action = AverageMeter('Action MSELoss', ':.4e')
            vals_action = AverageMeter('Action MSELoss', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, losses_total, vals_action],
                                     prefix="Epoch: [{}]".format(epoch))
        
        self.model.train()

        end = time.time()
        for i, data in enumerate(self.loader):
            target = {}
            target_action = th.squeeze(data['next_action'])
            target_value = th.squeeze(data['accumulated_reward'])
            target['action'] = target_action
            target['accumulated_reward'] = target_value
            # if self.config.use_reward:
            #     target_reward = th.squeeze(data['next_reward'])
            #     target['reward'] = target_reward

            losses = {}
            if self.config.model == 'CVAE':
                recon_x, mean, log_var, z = self.model(data)
                loss = self.loss_fn(recon_x, target['action'], mean, log_var)

                losses_elbo.update(loss['total'].item(), data['observation'].size(0))
                losses_recon.update(loss['Recon'].item(), data['observation'].size(0))
                losses_kld.update(loss['KL_div'].item(), data['observation'].size(0))
            
                losses['total'] = losses_elbo.avg
                losses['Recon'] = losses_recon.avg
                losses['KL_div'] = losses_kld.avg

            elif self.config.model == 'ValueNet':
                pred = self.model(data)
                loss = self.loss_fn(pred, target['accumulated_reward'])

                losses_total.update(loss['total'].item(), data['observation'].size(0))
                losses['total'] = losses_total.avg

            elif self.config.model == 'PolicyValueNet':
                value, recon_x, mean, log_var, z = self.model(data)
                loss = self.loss_fn(value, recon_x, target, mean, log_var)

                losses_total.updata(loss['total'].item(), data['observation'].size(0))
                losses_elbo.update(loss['ELBO'].item(), data['observation'].size(0))
                losses_recon.update(loss['Recon'].item(), data['observation'].size(0))
                losses_kld.update(loss['KL_div'].item(), data['observation'].size(0))
                losses_mse.update(loss['MSE'].item(), data['observation'].size(0))
            
                losses['total'] = losses_total.avg
                losses['ELBO'] = losses_elbo.avg
                losses['Recon'] = losses_recon.avg
                losses['KL_div'] = losses_kld.avg
                losses['MSE'] = losses_mse.avg

            else:
                pred = self.model(data)
                loss = self.loss_fn(pred, target)
                
                losses_total.update(loss['total'].item(), data['observation'].size(0))
                losses_action.update(loss['action'].item(), data['observation'].size(0))
                # if self.config.use_reward:
                #     losses_reward.update(loss['reward'].item(), data['observation'].size(0))

                losses['total'] = losses_total.avg
                losses['action'] = losses_action.avg
                # if self.config.use_reward:
                #     losses['reward'] = losses_reward.avg

            # Backprop + Optimize ...
            self.optim.zero_grad()
            loss['total'].backward()
            # th.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optim.step()

            if self.scheduler is not None:
                self.scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config.train_eval_freq == 0:
                self.model.eval()

                with th.no_grad():
                    vals = {}
                    if self.config.model == 'CVAE':
                        val = self.eval_fn(recon_x, target['action'], mean, log_var)
                        
                        vals_elbo.update(val['total'].item(), data['observation'].size(0))
                        vals_recon.update(val['Recon'].item(), data['observation'].size(0))
                        vals_kld.update(val['KL_div'].item(), data['observation'].size(0))
                
                        vals['total'] = vals_elbo.avg
                        vals['Recon'] = vals_recon.avg
                        vals['KL_div'] = vals_kld.avg
                    
                    elif self.config.model == 'ValueNet':
                        val = self.eval_fn(pred, target['accumulated_reward'])
                        
                        vals_total.update(val['total'].item(), data['observation'].size(0))                
                        vals['total'] = vals_total.avg

                    elif self.config.model == 'PolicyValueNet':
                        val = self.eval_fn(value, recon_x, target, mean, log_var)

                        vals_total.updata(val['total'].item(), data['observation'].size(0))
                        vals_elbo.update(val['ELBO'].item(), data['observation'].size(0))
                        vals_recon.update(val['Recon'].item(), data['observation'].size(0))
                        vals_kld.update(val['KL_div'].item(), data['observation'].size(0))
                        vals_mse.update(val['MSE'].item(), data['observation'].size(0))
                    
                        vals['total'] = vals_total.avg
                        vals['ELBO'] = vals_elbo.avg
                        vals['Recon'] = vals_recon.avg
                        vals['KL_div'] = vals_kld.avg
                        vals['MSE'] = vals_mse.avg

                    else:
                        val = self.eval_fn(pred, target)
                
                        vals_action.update(val['action'].item(), data['observation'].size(0))
                        # if self.config.use_reward:
                        #     vals_reward.update(val['reward'].item(), data['observation'].size(0))

                        vals['action'] = vals_action.avg
                        # if self.config.use_reward:
                        #     vals['reward'] = vals_reward.avg
                
                self.model.train()

            if i % self.config.print_freq == 0:
                progress.display(i)
 
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
