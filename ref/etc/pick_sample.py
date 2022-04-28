import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from dataclasses import dataclass, replace
from simple_parsing import Serializable

from model import GPT2, RNN, LSTM, CVAE
from saver import load_checkpoint
from utils import CosineAnnealingWarmUpRestarts
from load import LightDarkDataset

from visualize_long import Settings, collect_data


config = Settings(model='CVAE', model_name='9.27_CVAE')

dataset_path = os.path.join(os.getcwd(), config.path)
dataset_filename = config.test_file
data_dir = os.path.join(os.getcwd(), 'Learning/dataset')
device = config.device

with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
    dataset = pickle.load(f)

dataset = LightDarkDataset(config, dataset)

data, targets = collect_data(config, dataset)

with open(os.path.join(data_dir, 'light_dark_sample_len15.pickle'), 'wb') as f:
    pickle.dump({'data': data, 'targets': targets}, f)

print(data)
print(len(targets))

targets = np.asarray(targets).reshape(-1, 2)

plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.scatter(targets[:,0], targets[:,1], c='red')
plt.show()