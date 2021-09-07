import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as th


path: str = 'Learning/dataset'
dataset_path = os.path.join(os.getcwd(), path)
dataset_filename = 'light_dark_test.pickle'

with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
    dataset = pickle.load(f)

for a in dataset['action']:
    print(a[1])
print()
for o in dataset['observation']:
    print(o[1])
