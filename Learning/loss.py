import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


class RegressionLoss(nn.Module):
    def __init__(self, config):
        super(RegressionLoss, self).__init__()
        self.config = config
        self.loss_action = nn.SmoothL1Loss()
        # if self.config.use_reward:
        #     self.coefficient_loss = config.coefficient_loss
        #     self.loss_reward = nn.MSELoss()
    
    def forward(self, pred, target):
        loss = {}
        loss_action = self.loss_action(pred['action'], target['action'])
        loss['action'] = loss_action
        
        # if self.config.use_reward:
        #     loss_reward = self.loss_reward(pred['reward'], target['reward'])
        #     loss['reward'] = loss_reward
        #     loss_total = loss_action + self.coefficient_loss * loss_reward
        # else:
        #     loss_total = loss_action

        # loss['total'] = loss_total
        loss['total'] = loss_action

        return loss


class ELBOLoss(nn.Module):
    def __init__(self, config):
        super(ELBOLoss, self).__init__()
        self.config = config

    def forward(self, recon_x, x, mean, log_var):
        loss = {}
        recon_loss = F.mse_loss(recon_x, x)
        kld = -0.5 * th.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0) # Using reparameterization
        loss['Recon'] = recon_loss
        loss['KL_div'] = kld
        loss['total'] = recon_loss + kld

        return loss