from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
import logging
import torch as th

from run import Settings


class Evaluator(Object):
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """
    def __init__(self,
                 config: Settings,
                 loader: th.utils.data.DataLoader,
                 model: th.nn.Module,
                 eval_fn: Callable = None
                 ):
        self.config = config
        self.loader = loader
        self.model = model
        self.eval_fn = eval_fn
    
    def eval(self):
        self.model.eval()
        with th.no_grad():
            for i, data in enumerate(self.loader):
                observations, actions, attn_mask = data['observations'], data['actions'], data['mask']
                # |FIXME|
                target_actions = th.clone(data['next_action'])

                pred_actions = self.model(observations, actions, attn_mask)
                out = self.eval_fn(pred_actions, target_actions)

                # |FIXME| Logging

                return out