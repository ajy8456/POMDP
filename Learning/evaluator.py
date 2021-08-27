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
                 eval_fn: Callable = None
                 ):
        self.config = config
        self.loader = loader
        self.model = model
        self.eval_fn = eval_fn
    
    def eval(self, epoch):        
        batch_time = AverageMeter('Time', ':6.3f')
        vals = AverageMeter('MAE', ':.4e')
        progress = ProgressMeter(len(self.loader),
                                 [batch_time, vals],
                                 prefix="Epoch: [{}]".format(epoch))
        
        self.model.eval()
        if str(self.config.device) == 'cuda':
            th.cuda.empty_cache()

        with th.no_grad():
            end = time.time()
            for i, data in enumerate(self.loader):
                observations, actions, time_steps, attn_mask = data['observation'], data['action'], data['timesteps'], data['mask']
                target_actions = data['next_action']

                pred_actions = self.model(observations, actions, time_steps, attn_mask)
                val = self.eval_fn(pred_actions, target_actions)

                # measure elapsed time
                vals.update(val.item(), data['observation'].size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

        return vals.avg
