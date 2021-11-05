import torch as th

a = th.tensor([[1,0], [0,1], [1,1]], dtype=th.float32)
b = th.tensor([[0,0], [2,0]], dtype=th.float32)
print(a.shape, b.shape)
print(a.size(0), b.size(0))

c = th.empty(a.size(0), b.size(0)).fill_(1e9)
print(c.size())
print(c)

for m in range(a.size(0)):
    for n in range(b.size(0)):
        c[m][n] = th.norm(a[m] - b[n], 2)
print(c.size(), c)
c_min = th.min(c)
print(c_min)