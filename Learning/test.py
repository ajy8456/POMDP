import os
import glob
import shutil
import pickle
import numpy as np
from simple_parsing.helpers.serialization.serializable import D
import torch as th
import matplotlib.pyplot as plt


# path: str = 'Learning/dataset'
# dataset_path = os.path.join(os.getcwd(), path)
# dataset_filename = 'light_dark_long_test_100K.pickle'

# with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
#     dataset = pickle.load(f)

# print(len(dataset['observation']))

# reward = dataset['reward']
# sum_reward = []
# for i in range(len(reward)):
#     sum_reward.append(np.sum(reward[i]))

# sorted_index = sorted(range(len(sum_reward)), key = lambda k: sum_reward[k])

# top_action, top_observation, top_next_state, top_reward, top_traj_len = [], [], [], [], []
# for idx in range(int(len(dataset['observation']) * 0.2)):
#     top_action.append(dataset['action'][idx])
#     top_observation.append(dataset['observation'][idx])
#     top_next_state.append(dataset['next_state'][idx])
#     top_reward.append(dataset['reward'][idx])
#     top_traj_len.append(dataset['traj_len'][idx])

# top_dataset = {'action': top_action,
#                'observation': top_observation,
#                'next_state': top_next_state,
#                'reward': top_reward,
#                'traj_len': top_traj_len}

# print(len(top_dataset['observation']))
# # print(top_dataset['traj_len'])

# with open(os.path.join(dataset_path, 'light_dark_long_test_100K_top20%_20K.pickle'), 'wb') as f:
#     pickle.dump(top_dataset, f)


# fig, ax = plt.subplots(1, 2)

# ax[0].hist(dataset['traj_len'])
# ax[1].hist(top_dataset['traj_len'])

# ax[0].plot(std, success_rate, c='green')
# ax[1].plot(std[1:], val_success[1:], label = 'success')
# ax[1].plot(std, val_fail, label = 'fail')
# ax[1].legend(loc='best')

# ax[0].title.set_text("Success Rate")
# ax[1].title.set_text("Value of Root Node")

# ax[0].set_xlabel('Deviation')
# ax[0].set_ylabel('Sucess Rate(%)')
# ax[1].set_xlabel('Deviation')
# ax[1].set_ylabel('Value of Root Node(Avg.)')

# plt.show()


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

# std = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# success_rate = np.array([0, 3, 18, 39, 25, 28, 14])
# val_success = np.array([None, 1.88, 3.90, 5.41, 4.89, 9.50, 3.77])
# val_fail = np.array([-4.17, -1.11, 1.63, 1.76, 0.65, -0.53, -1.45])


# num_sim = np.array([10, 100, 1000, 10000])
# success_rate_exec = np.array([[50, 58, 71, 65], [67, 64, 52, 49]])
# success_rate_sim = np.array([[7.85, 10.88, 16.02, 23.11], [10.72, 13.20, 24.95, 28.94]])
# node_val = np.array([[3.31, 3.27, 10.07, 18.83], [14.19, 10.34, 22.57, 24.62]])

# fig, ax = plt.subplots(1, 3)

# ax[0].semilogx(num_sim, success_rate_exec[0], label='unguided')
# ax[0].semilogx(num_sim, success_rate_exec[1], label='guided')
# ax[0].legend()
# ax[1].semilogx(num_sim, success_rate_sim[0], label='unguided')
# ax[1].semilogx(num_sim, success_rate_sim[1], label='guided')
# ax[1].legend()
# ax[2].semilogx(num_sim, node_val[0], label='unguided')
# ax[2].semilogx(num_sim, node_val[1], label='guided')
# ax[2].legend()

# ax[0].title.set_text("Success Rate(Execution)")
# ax[1].title.set_text("Success Rate(Simulation)")
# ax[2].title.set_text("Value of Root Node(Averge)")

# ax[0].set_xlabel('# simulation')
# ax[0].set_ylabel('(%)')
# ax[1].set_xlabel('# simulation')
# ax[1].set_ylabel('(%)')
# ax[2].set_xlabel('# simulation')
# ax[2].set_ylabel('(Avg.)')

# plt.show()

# total_reward = np.asarray([38.91, 41.11, 43.74, 44.59, 48.40, 52.01])
# success_rate = np.asarray([43, 50, 55, 61, 64, 70])

# plt.plot(success_rate, total_reward)
# # plt.savefig('total_reward-success_rate.png')
# plt.show()


num_sim = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000])
success_rate_exec = np.array([[47.9, 84.1, 83.0, 80.9, 83.3, 81.6, 79.4, 75.9, 74.7, 70.5, 73.3, 51.2, 35.4, 33.0],
                              [10.6, 48.2, 54.9, 56.0, 61.3, 61.8, 67.3, 65.5, 65.3, 68.4, 67.6, 72.1, 78.0, 74.5]])
plt.plot(num_sim, success_rate_exec[0], label='Guided')
plt.plot(num_sim, success_rate_exec[1], label='Unguided')
# plt.xscale('log')
plt.legend()
plt.show()


# path = os.path.join(os.getcwd(), 'Learning/dataset/mcts_1')
# # print(path)
# file_list = glob.glob(path + '/*')
# # print(file_list)
# # print(len(file_list))
# file_list = [file for file in file_list if file.endswith(".pickle")]
# # print(len(file_list))

# for file in file_list:
#     shutil.move(file, path)
    
    
# dataset_path = os.path.join(os.getcwd(), 'Learning/dataset')
# filename = 'mcts_1_train'
# dataset = glob.glob(f'{dataset_path}/{filename}/*.pickle')
# total_reward_success = []
# total_reward_fail = []
# for data in dataset:
#     with open(data, 'rb') as f:
#         traj = pickle.load(f)
#         if traj[-1] > 50:
#             total_reward_success.append(traj[-1])
#         else:
#             total_reward_fail.append(traj[-1])
# print(np.asarray(total_reward_success).min())
# print(np.asarray(total_reward_success).max())
# print(np.asarray(total_reward_fail).min())
# print(np.asarray(total_reward_fail).max())
# # plt.hist(total_reward, bins=110, range=(-10, 100))
# # plt.savefig('total_reward.png')