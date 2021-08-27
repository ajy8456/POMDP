import os
import time
from dataclasses import dataclass, replace
from simple_parsing import Serializable
import pickle
import torch as th
from torch.utils.tensorboard import SummaryWriter

from load import get_loader
from model import GPT2
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import CosineAnnealingWarmUpRestarts


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    batch_size: int = 1024
    shuffle: bool = True
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1

    # Architecture
    dim_embed: int = 128
    dim_hidden: int = 128
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4

    num_layers: int = 3

    train_pos_en: bool = True

    dropout: float = 0.0
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
    resume: str = None # checkpoint file name for resuming
    # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
    epochs: int = 10000

    # Learning rate
    # |NOTE| using small learning rate, in order to apply warm up
    learning_rate: float = 1e-7
    weight_decay: float = 1e-4
    warmup_step: int = int(1e4)
    # For cosine annealing
    T_0: int = int(1e4)
    T_mult: int = 2
    lr_max: float = 0.1
    lr_mult: float = 0.9

    # Logging
    exp_dir: str = 'Learning/exp'
    model_name: str = '8.27_batch1024_maxlen100_dim128_layer3_AdamWR_TrainPosEn'
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 1000 # per epochs


def main():
    config = Settings()
    # |TODO| go to Setting()
    train_filename = 'light_dark_10K.pickle'
    test_filename = 'light_dark_tiny.pickle'
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
    model = GPT2(config).to(device)
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    # scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/config.warmup_step, 1))
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer=optimizer,
        T_0=config.T_0,
        T_mult=config.T_mult,
        eta_max=config.lr_max,
        T_up=config.warmup_step,
        gamma=config.lr_mult
    )
    loss_fn = th.nn.SmoothL1Loss()
    eval_fn = th.nn.L1Loss()

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
            print("No checkpoint found at '{}'".format(config.resume))
            return


    for epoch in range(start_epoch, config.epochs+1):
        print(f'===== Start {epoch} epoch =====')
        
        # Training one epoch
        print("Training...")
        train_loss, train_val = trainer.train(epoch)
        logger.add_scalar('Loss/train', train_loss, epoch)
        logger.add_scalar('Eval/train', train_val, epoch)

        # evaluating
        if epoch % config.test_eval_freq == 0:
            print("Validating...")
            test_val = evaluator.eval(epoch)

            # save the best model
            if test_val < best_error:
                best_error = test_val

                save_checkpoint('Saving the best model!',
                                os.path.join(model_dir, 'best.pth'),
                                epoch, 
                                best_error, 
                                model, 
                                optimizer, 
                                scheduler
                                )
            logger.add_scalar('Eval/test', test_val, epoch)
        
        # save the model
        if epoch % config.save_freq == 0:
            save_checkpoint('Saving...', 
                            os.path.join(model_dir, f'ckpt_epoch_{epoch}.pth'), 
                            epoch, 
                            test_val, 
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