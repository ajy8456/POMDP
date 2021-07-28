from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
import logging
import torch as th

from run import Settings


class Trainer(object):
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """
    def __init__(self,
                 config: Settings,
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

    def _train(self, step):
        """Internal function for dealing with the inner training loop."""
        for i, data in enumerate(self.loader):
            observations, actions, attn_mask = data['observations'], data['actions'], data['mask']
            # |FIXME|
            target_actions = th.clone(actions)

            pred_actions = self.model(observations, actions, attn_mask)

            loss = self.loss_fn(pred_actions, target_actions)
            # |FIXME| Logging

            # Backprop + Optimize ...
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            step += 1

            if step % self.config.eval_freq == 0:
                eval_val = self.eval_fn()
                # |FIXME| Logging

            if step >= self.config.train_steps:
                return step

    def train(self, step):
        self.model.train()
        try:
            self._train(step)
        except KeyboardInterrupt:
            logging.info('Terminating training due to SIGINT')
        # finally:
        #     # TODO(ycho): When an interrupt occurs, the current state
        #     # will ALWAYS be saved to a hardcoded file in a temporary directory.
        #     # Maybe this is a bad idea.
        #     Saver(self.model, self.optim).save('/tmp/model-backup.zip')

    def eval(self):
        return