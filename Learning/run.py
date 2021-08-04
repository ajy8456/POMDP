import os
from dataclasses import dataclass, replace
from simple_parsing import Serializable
import pickle
import torch as th
from torch.utils.tensorboard import SummaryWriter

from load import get_loader
from model import GPT2
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    batch_size: int = 128
    shuffle: bool = True
    max_len: int = 1000
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

    dropout: float = 0.0
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
    # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
    epochs: int = 100
    learning_rate: float = 2 * 1e-6

    # Logging
    exp_dir: str = 'Learning/exp'
    model_name: str = '2_lr21e-6'
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 20 # per epochs


def main():
    config = Settings()
    # |TODO| go to Setting()
    train_filename = 'light_dark_train.pickle'
    test_filename = 'light_dark_test.pickle'
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
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = th.nn.SmoothL1Loss()
    eval_fn = th.nn.L1Loss()

    # Trainer & Evaluator
    trainer = Trainer(config=config,
                      loader=train_loader,
                      model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      eval_fn=eval_fn)
    evaluator = Evaluator(config=config,
                          loader=test_loader,
                          model=model,
                          eval_fn=eval_fn)

    best_error = 10000.
    for epoch in range(1, config.epochs+1):
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

                save_checkpoint(epoch, model, optimizer, best_error,
                                os.path.join(model_dir, 'best.pth'),
                                'Saving the best model!')
            logger.add_scalar('Eval/test', test_val, epoch)
        
        # save the model
        if epoch % config.save_freq == 0:
            save_checkpoint(epoch, model, optimizer, test_val,
                            os.path.join(model_dir, f'ckpt_epoch_{epoch}.pth'),
                            'Saving...')

        print(f'===== End {epoch} epoch =====')


if __name__ == '__main__':
    main()