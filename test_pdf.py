import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.001, 0.999, 1000)

z_i = np.log(-1-(1/(x-1)))

mu = 0
sigma = [0.01, 0.1, 1, 2, 3]

p_x = []
for sig in sigma:
    p_x.append(np.multiply(np.power(np.exp(z_i/2) + (np.exp(-z_i/2)), 2), (1/np.sqrt(2*np.pi*sig)) * np.exp(-np.power(z_i - mu, 2) / (2*sig**2))))

for i in range(len(sigma)):
    plt.plot(x, p_x[i], label=f"std={sigma[i]}")

plt.xlabel("X")
plt.ylabel("P(X)")
plt.legend()

plt.show()