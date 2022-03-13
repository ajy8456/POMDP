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
# filename: str = 'sim_success_mcts1_exp_const_30_std0.5' # folder name
# dataset = glob.glob(f'{dataset_path}/{filename}/*.pickle')
# for d in dataset:
#     if os.path.getsize(d) == 0:
#         print(d)
# d = dataset[14536]
# with open(d, 'rb') as f:
#     data = pickle.load(f)
# print(data)

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


num_sim = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])
# success_rate_exec = np.array([[46.2, 52.8, 57.1, 61.7, 62.0, 63.5, 63.8, 64.6, 65.5, 63.1, 66.5],
#                               [74.5, 76.2, 76.1, 75.6, 73.1, 71.8, 70.9, 68.3, 69.8, 70.7, 54.7],
#                               [64.2, 70.8, 71.7, 73.7, 73.4, 72.5, 71.6, 70.8, 71.2, 70.2, 61.0]])
# num_sim = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000])
success_rate_exec1 = np.array([[10.6, 28.2, 34.9, 46.0, 61.3, 61.8, 67.3, 65.5, 65.3, 66.1, 67.6, 72.1, 78.0, 74.5],
                               [24, 31, 39, 47, 50, 55, 61, 66, 70, 73.3, 88, 91, 94, 95]])
success_rate_exec2 = np.array([[60.7, 66.4, 70.6, 72.7, 75.7, 72.9, 74.6, 74.4, 73.8, 71.0, 74, 53, 55, 49],
                               [57.8, 68.8, 74.7, 80.8, 84.0, 85.6, 88.3, 91.1, 90.3, 90.9, 97.0, 97.3, 97.8, 99]])
# diff1 = success_rate_exec1[1] - success_rate_exec1[0]
# diff2 = success_rate_exec2[1] - success_rate_exec2[0]
plt.plot(num_sim, success_rate_exec1[0], label='Unguided(before)', color='red', linestyle='--')
plt.plot(num_sim, success_rate_exec1[1], label='Unguided(after)', color='red')
plt.plot(num_sim, success_rate_exec2[0], label='Guided by scheme1+1+1(before)', color='blue', linestyle='--')
plt.plot(num_sim, success_rate_exec2[1], label='Guided by scheme1+1+1(after)', color='blue')
# plt.plot(num_sim, diff1, label='Before')
# plt.plot(num_sim, diff2, label='After')

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

# x = np.asarray([0.01, 0.02, 0.02, 0.05, 0.05, 0.03, 0.06, 0.08, 0.10, 0.10, 0.11, 0.12, 0.11, 0.13, 0.14, 0.14, 0.15, 0.17, 0.18, 0.18, 0.19, 0.20, 0.17, 0.21, 0.22, 0.24, 0.24, 0.26, 0.27, 0.28, 0.31, 0.32, 0.32, 0.34, 0.33, 0.35, 0.36, 0.37, 0.39, 0.36, 0.35, 0.37, 0.35, 0.35, 0.38, 0.37, 0.37, 0.41, 0.47, 0.50])
# print(len(x))
# y = np.asarray([30.13, 34.52, 33.51, 36.18, 36.51, 31.13, 29.10, 28.59, 28.69, 22.41, 20.14, 28.14, 19.52, 10.52, 24.51, 21.52, 1.50, 15.69, 8.50, 1.50, 18.02, 15.05, 18.69, 14.60, 15.07, -2.69, -8.59, -7.69, -3.73, -20.60, -18.52, -24.49, -21.50, -23.59, -21.52, -20.59, -18.38, -17.28, -19.39, -19.40, -17.50, -16.70, -18.48, -22.40, -15.25, -15.39, -15.20, -21.20, -21.50, -24.39])
# print(len(y))
# plt.scatter(x, y)
# plt.show()