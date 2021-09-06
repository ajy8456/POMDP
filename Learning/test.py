import pickle
import torch as th

n=2
data = []
while n <= 32:
    data.append(th.arange(1, n+1))
    n += 1

print(len(data))
print(data)

with open('data_test.pickle', 'wb') as f:
    pickle.dump(data, f)