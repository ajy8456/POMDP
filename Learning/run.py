import os
import time
from dataclasses import dataclass, replace
from simple_parsing import Serializable
import pickle
import torch as th
from tensorboardX import SummaryWriter

from load import get_loader
from model import GPT2, RNN, LSTM
from loss import RegressionLoss
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    train_file: str = 'light_dark_10K.pickle'
    test_file: str = 'light_dark_10K.pickle'
    batch_size: int = 4096 # 100steps/epoch
    shuffle: bool = True # for using Sampler, it should be False
    use_sampler: bool = False
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1

    # Architecture
    model: str = 'LSTM' # GPT or RNN or LSTM
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 128
    dim_hidden: int = 128
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4

    num_layers: int = 3

    train_pos_en: bool = False
    use_reward: bool = True
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.0
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
    resume: str = None # checkpoint file name for resuming
    # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
    epochs: int = 100

    # Learning rate
    # |NOTE| using small learning rate, in order to apply warm up
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_step: int = int(1e3)
    # For cosine annealing
    T_0: int = int(1e4)
    T_mult: int = 1
    lr_max: float = 0.01
    lr_mult: float = 0.5

    # Logging
    exp_dir: str = 'Learning/exp'
    model_name: str = '9.14_10Kdata_LSTM_log_eff_grad'
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 10 # per epochs
    save_freq: int = 100 # per epochs

    print_in_out: bool = False


def main():
    config = Settings()
    # |TODO| go to Setting()
    train_filename = config.train_file
    test_filename = config.test_file
    dataset_path = os.path.join(os.getcwd(), config.path)
    
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)

    logger = SummaryWriter(model_dir)

    with open(os.path.join(dataset_path, train_filename), 'rb') as f:
        train_dataset = pickle.load(f)
    with open(os.path.join(dataset_path, test_filename), 'rb') as f:
        test_dataset = pickle.load(f)
    
    print('#trajectories of train_dataset:', len(train_dataset['observation']))
    print('#trajectories of test_dataset:', len(test_dataset['observation']))

    # generate dataloader
    train_loader = get_loader(config, train_dataset)
    test_loader = get_loader(config, test_dataset)

    # model
    device = th.device(config.device)
    if config.model == 'GPT':
        model = GPT2(config).to(device)
    elif config.model == 'RNN':
        model = RNN(config).to(device)
    elif config.model == 'LSTM':
        model = LSTM(config).to(device)
    else:
        raise Exception(f'"{config.model}" is not support!! You should select "GPT", "RNN", or "LSTM".')

    # optimizer
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    # learning rate scheduler
    if config.optimizer == 'AdamW':
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/config.warmup_step, 1))
    elif config.optimizer == 'AdamWR':
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_max=config.lr_max,
            T_up=config.warmup_step,
            gamma=config.lr_mult
        )
    else:
        raise Exception(f'"{config.optimizer}" is not support!! You should select "AdamW" or "AdamWR".')

    # Metric
    loss_fn = RegressionLoss(config)
    eval_fn = RegressionLoss(config)

    # Trainer & Evaluator
    trainer = Trainer(config=config,
                      loader=train_loader,
                      model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      loss_fn=loss_fn,
                      eval_fn=eval_fn)
    evaluator = Evaluator(config=config,
                          loader=test_loader,
                          model=model,
                          eval_fn=eval_fn)

    # save configuration
    config.save(model_dir + '/config.yaml')
    # Logging model graph
    dummy = next(iter(test_loader))
    for k in dummy:
        dummy[k].to(device).detach()
    logger.add_graph(ModelAsTuple(model), dummy)

    start_epoch = 1
    best_error = 10000.

    # load checkpoint for resuming
    if config.resume is not None:
        filename = os.path.join(model_dir, config.resume)
        if os.path.isfile(filename):
            start_epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, filename, model, optimizer, scheduler)
            start_epoch += 1
            print("Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch))
        else:
            raise Exception("No checkpoint found at '{}'".format(config.resume))

    for epoch in range(start_epoch, config.epochs+1):
        print(f'===== Start {epoch} epoch =====')
        
        # Training one epoch
        print("Training...")
        train_loss, train_val = trainer.train(epoch)

        # Logging
        logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
        logger.add_scalar('Loss(action)/train', train_loss['action'], epoch)
        log_gradients(model, logger, epoch, save_grad=True, save_param=True)
        # if config.use_reward:
        #     logger.add_scalar('Loss(reward)/train', train_loss['reward'], epoch)

        logger.add_scalar('Eval(action)/train', train_val['action'], epoch)
        # if config.use_reward:
        #     logger.add_scalar('Eval(reward)/train', train_val['reward'], epoch)

        # evaluating
        if epoch % config.test_eval_freq == 0:
            print("Validating...")
            test_val = evaluator.eval(epoch)

            # save the best model
            if test_val['action'] < best_error:
                best_error = test_val['action']

                save_checkpoint('Saving the best model!',
                                os.path.join(model_dir, 'best.pth'),
                                epoch, 
                                best_error, 
                                model, 
                                optimizer, 
                                scheduler
                                )
            
            # Logging
            logger.add_scalar('Eval(action)/test', test_val['action'], epoch)
            # if config.use_reward:
            #     logger.add_scalar('Eval(reward)/test', test_val['reward'], epoch)
        
        # save the model
        if epoch % config.save_freq == 0:
            save_checkpoint('Saving...', 
                            os.path.join(model_dir, f'ckpt_epoch_{epoch}.pth'), 
                            epoch, 
                            test_val['action'], 
                            model, 
                            optimizer, 
                            scheduler
                            )

        print(f'===== End {epoch} epoch =====')


if __name__ == '__main__':
    total_time_start = time.time()
    main()
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print("Total Time:", total_time)