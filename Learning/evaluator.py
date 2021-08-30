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
        vals_action = AverageMeter('Action MAE', ':.4e')
        vals_reward = AverageMeter('Reward MSE', ':.4e')
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
                target['action'] = target_action
                if self.config.use_reward:
                    target_reward = th.squeeze(data['next_reward'])
                    target['reward'] = target_reward

                pred = self.model(data)

                val = self.eval_fn(pred, target)

                # measure elapsed time
                vals_action.update(val['action'].item(), data['observation'].size(0))
                if self.config.use_reward:
                    vals_reward.update(val['reward'].item(), data['observation'].size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

            vals = {}
            vals['action'] = vals_action.avg
            if self.config.use_reward:
                vals['reward'] = vals_reward.avg

        return vals
