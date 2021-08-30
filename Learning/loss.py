import torch as th
import torch.nn as nn

from typing import Dict


class RegressionLoss(nn.Module):
    def __init__(self, config):
        super(RegressionLoss, self).__init__()
        self.config = config
        self.loss_action = nn.SmoothL1Loss()
        if self.config.use_reward:
            self.coefficient_loss = config.coefficient_loss
            self.loss_reward = nn.MSELoss()
    
    def forward(self, pred, target):
        loss_action = self.loss_action(pred['action'], target['action'])
        if self.config.use_reward:
            loss_reward = self.loss_reward(pred['reward'], target['reward'])

        loss_total = loss_action + self.coefficient_loss * loss_reward

        loss = {'total': loss_total, 'action': loss_action, 'reward': loss_reward}

        return loss