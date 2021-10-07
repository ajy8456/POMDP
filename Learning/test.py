import os
import pickle
import numpy as np
from simple_parsing.helpers.serialization.serializable import D
import torch as th
import matplotlib.pyplot as plt


# path: str = 'Learning/dataset'
# dataset_path = os.path.join(os.getcwd(), path)
# dataset_filename = 'light_dark_long_mini.pickle'

# with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
#     dataset = pickle.load(f)

# print(len(dataset['observation']))

# observation, action, reward, next_state = [], [], [], []
# indices = np.random.choice(len(dataset['observation']), 4)
# for idx in indices:
#     observation.append(dataset['observation'][idx])
#     action.append(dataset['action'][idx])
#     reward.append(dataset['reward'][idx])
#     next_state.append(dataset['next_state'][idx])
# tiny_data = {
#     'observation': observation,
#     'action': action,
#     'reward': reward,
#     'next_state': next_state
# }
# with open(dataset_path + '/super_tiny_dataset.pickle', 'wb') as f:
#     pickle.dump(tiny_data, f)


# n=2
# data = []
# while n <= 32:
#     data.append(th.arange(1, n+1))
#     n += 1

# print(len(data))
# print(data)

# with open('data_test.pickle', 'wb') as f:
#     pickle.dump(data, f)


# with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
#     dataset = pickle.load(f)

# index = np.random.randint(len(dataset['observation']))
# print(len(dataset['observation']))
# print(index)

# observation = dataset['observation'][index]
# action = dataset['action'][index]
# reward = dataset['reward'][index]
# next_state = dataset['next_state'][index]
# traj_len = dataset['traj_len'][index]
# traj = {'observation': observation,
#             'action': action,
#             'reward': reward,
#             'next_state': next_state,
#             'traj_len': traj_len}

std = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
success_rate = np.array([0, 3, 18, 39, 25, 28, 14])
val_success = np.array([None, 1.88, 3.90, 5.41, 4.89, 9.50, 3.77])
val_fail = np.array([-4.17, -1.11, 1.63, 1.76, 0.65, -0.53, -1.45])

fig, ax = plt.subplots(1, 2)

ax[0].plot(std, success_rate, c='green')
ax[1].plot(std[1:], val_success[1:], label = 'success')
ax[1].plot(std, val_fail, label = 'fail')
ax[1].legend(loc='best')

ax[0].title.set_text("Success Rate")
ax[1].title.set_text("Value of Root Node")

ax[0].set_xlabel('Deviation')
ax[0].set_ylabel('Sucess Rate(%)')
ax[1].set_xlabel('Deviation')
ax[1].set_ylabel('Value of Root Node(Avg.)')

plt.show()