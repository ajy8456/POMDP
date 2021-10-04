import math
import torch as th
from torch.optim.lr_scheduler import _LRScheduler


class ModelAsTuple(th.nn.Module):
    """Workaround to avoid tracing bugs in add_graph from rejecting outputs of form Dict[Schema,Any]."""
    def __init__(self, config, model: th.nn.Module):
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, inputs):
        if self.config.model == 'CVAE':
            return tuple(v[0] for v in self.model(inputs))
        return tuple(v for (k, v) in self.model(inputs).items())


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    '''
    Custom CosineAnnelingWarmUpRestarts, appending usage of adjusting warm up start & max value
    Code Reference: https://github.com/gaussian37/pytorch_deep_learning_models/blob/master/cosine_annealing_with_warmup/cosine_annealing_with_warmup.py
    '''
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        '''
        Args:
            T_0: initial period
            T_mult: how much multiplying to period after each period
            eta_max: maximum of learning rate
            T_up: how many steps for warm up
            gamma: how much multiplying to eta_max after each period
        '''
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def log_gradients(model, logger, step, log_grad, log_param, eff_grad, print_num_para):
    
    cnt_para = 0
    
    for tag, value in model.named_parameters():
        if value.grad is not None:

            cnt_para += th.numel(value.grad)
            
            if log_grad:
                if eff_grad:
                    filt_1 = th.where(value.grad>1e-5, True, False)
                    filt_2 = th.where(value.grad<-1e-5, True, False)
                    filt = th.logical_or(filt_1, filt_2)
                    eff_grad = value.grad[filt].clone().cpu()
                    if eff_grad.tolist():
                        logger.add_histogram(tag + "/grad", eff_grad, step)
                    else:
                        # |FIXME| How to log no data?
                        logger.add_histogram(tag + "/grad", th.zeros(1).cpu(), step)
                else:
                    logger.add_histogram(tag + "/grad", value.grad.clone().cpu(), step)

            if log_param:
                logger.add_histogram(tag + "/param", value.clone().cpu(), step)
            
            if print_num_para:
                print(f"#Trainable parameters of {tag}", th.numel(value.grad))
    if print_num_para:
        print("#Total trainable parameters:", cnt_para)