import time
import torch as th
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)

from trainer import AverageMeter, ProgressMeter

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
        if self.config.model == 'CVAE':
            vals_elbo = AverageMeter('ELBO', ':4e')
            vals_recon = AverageMeter('Reconstruction Error', ':4e')
            vals_kld = AverageMeter('KL-divergence', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, vals_elbo],
                                     prefix="Epoch: [{}]".format(epoch))

        elif self.config.model == 'ValueNet':
            vals_total = AverageMeter('MSELoss', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, vals_total],
                                     prefix="Epoch: [{}]".format(epoch))

        elif self.config.model == 'PolicyValueNet':
            # |TODO| implement Chamfer distance
            vals_total = AverageMeter('Total Eval', ':.4e')
            vals_elbo = AverageMeter('Action ELBOLoss', ':.4e')
            vals_recon = AverageMeter('Action Reconstruction Error', ':.4e')
            vals_kld = AverageMeter('Action KL-divergence', ':.4e')
            vals_mse = AverageMeter('Accumulated Reward MSELoss', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, vals_total],
                                     prefix="Epoch: [{}]".format(epoch))

        else:
            vals_action = AverageMeter('Action SmoothL1Loss', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, vals_action],
                                     prefix="Epoch: [{}]".format(epoch))
        
        self.model.eval()
        if str(self.config.device) == 'cuda':
            th.cuda.empty_cache()

        with th.no_grad():
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

                vals = {}
                if self.config.model == 'CVAE':
                    recon_x, mean, log_var, z = self.model(data)
                    val = self.eval_fn(recon_x, target['action'], mean, log_var)

                    vals_elbo.update(val['total'].item(), data['observation'].size(0))
                    vals_recon.update(val['Recon'].item(), data['observation'].size(0))
                    vals_kld.update(val['KL_div'].item(), data['observation'].size(0))
                
                    vals['total'] = vals_elbo.avg
                    vals['Recon'] = vals_recon.avg
                    vals['KL_div'] = vals_kld.avg

                elif self.config.model == 'ValueNet':
                    pred = self.model(data)
                    val = self.eval_fn(pred, target['accumulated_reward'])
                    
                    vals_total.update(val['total'].item(), data['observation'].size(0))                
                    vals['total'] = vals_total.avg

                elif self.config.model == 'PolicyValueNet':
                    value, recon_x, mean, log_var, z = self.model(data)
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
                    pred = self.model(data)
                    val = self.eval_fn(pred, target)

                    vals_action.update(val['action'].item(), data['observation'].size(0))
                    # if self.config.use_reward:
                    #     vals_reward.update(val['reward'].item(), data['observation'].size(0))
                    
                    vals['action'] = vals_action.avg
                    # if self.config.use_reward:
                    #     vals['reward'] = vals_reward.avg
            
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

        return vals
