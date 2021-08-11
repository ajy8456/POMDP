import os
from dataclasses import dataclass, replace
from simple_parsing import Serializable
import pickle
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from load import get_loader
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    batch_size: int = 1
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
    learning_rate: float = 1e-5

    # Logging
    exp_dir: str = 'Learning/exp'
    model_name: str = 'test_2_RNN'
    print_freq: int = 1000 # per train_steps
    train_eval_freq: int = 1000 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 20 # per epochs


class Test_2(nn.Module):
    def __init__(self, config):
        super(Test_2, self).__init__()
        self.config = config
        self.dim_observation = config.dim_observation
        self.dim_action = config.dim_action
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers


        self.embed = nn.Linear(self.dim_observation + self.dim_action, self.dim_embed)

        self.rnn = nn.RNN(input_size=self.dim_embed, hidden_size=self.dim_hidden, num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.dim_hidden, self.dim_action)


    def forward(self, observations, actions, attn_mask=None):
        batch_size, seq_len = observations.shape[0], observations.shape[1]
        
        input = th.cat((observations, actions), dim=-1)
        input_embeddings = self.embed(input)

        stacked_attention_mask = th.unsqueeze(attn_mask, dim=-1)
        stacked_attention_mask = th.repeat_interleave(~stacked_attention_mask, self.dim_hidden, dim=-1)
        input_embeddings.masked_fill_(stacked_attention_mask, 0)

        output, h_n = self.rnn(input_embeddings)
        x = self.fc1(h_n)

        return x


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
    model = Test_2(config).to(device)
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