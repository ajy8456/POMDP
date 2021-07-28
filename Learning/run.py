import os
from dataclasses import dataclass, replace
from simple_parsing import Serializable
import pickle
import torch as th

from load import get_loader
from model import GPT2
from trainer import Trainer


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'dataset'
    batch_size: int = 64
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
    train_steps: int = int(1e4)
    # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
    epochs: int = int(100)


opt = Settings()
# |TODO| go to Setting()
train_filename = 'light_dark_train.pickle'
test_filename = 'light_dark_test.pickle'
dataset_path = os.path.join(os.getcwd(), opt.path)

with open(os.path.join(dataset_path, train_filename), 'rb') as f:
    train_dataset = pickle.load(f)
with open(os.path.join(dataset_path, test_filename), 'rb') as f:
    test_dataset = pickle.load(f)

# append trajectory length to dataset
train_traj_lens = []
for traj in train_dataset:
    train_traj_lens.append(len(traj['observation']))
train_dataset['traj_lens'] = train_traj_lens

test_traj_lens = []
for traj in test_dataset:
    test_traj_lens.append(len(traj['observation']))
test_dataset['traj_lens'] = test_traj_lens

# generate dataloader
train_loader = get_loader(opt, train_dataset)
test_loader = get_loader(opt, test_dataset)