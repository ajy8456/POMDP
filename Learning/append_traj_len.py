import os
import pickle


load_dir = os.getcwd() + '/Learning/dataset'
save_dir = os.getcwd() + '/Learning/new_dataset'
filename = os.listdir(load_dir)

for file in filename:
    file = 'light_dark_10K.pickle'
    print(f'continuing....{file}')
    with open(os.path.join(load_dir, file), 'rb') as f:
        dataset = pickle.load(f)

    # traj_len = []
    # for d in dataset['observation']:
    #     traj_len.append(d.shape[0])
    # dataset['traj_len'] = traj_len

    p_sample = [t / sum(dataset['traj_len']) for t in dataset['traj_len']]
    dataset['p_sample'] = p_sample

    with open(os.path.join(save_dir, file), 'wb') as f:
        pickle.dump(dataset, f)