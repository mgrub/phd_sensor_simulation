import numpy as np
import matplotlib.pyplot as plt

N = 100000
nbins = min(N//100, 100)
x = np.random.normal(2.0, 2.0, size=N)
y = np.random.normal(2.5, 1.0, size=N)

fig, ax = plt.subplots(1, 1)

ax.hist(x, density=True, bins=nbins, alpha=0.5)
ax.hist(y, density=True, bins=nbins, alpha=0.5)
ax.hist(x*y, density=True, bins=nbins, alpha=0.8)

plt.show()