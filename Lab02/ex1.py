import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
nr_samples = 10000
m1 = np.random.exponential(scale=1/4, size=int(nr_samples * 0.4))
m2 = np.random.exponential(scale=1/6, size=int(nr_samples * (1 - 0.4)))
X = np.concatenate((m1, m2))
media = np.mean(X)
dev_std = np.std(X)
print("Media:",media)
print("Deviatia standard:",dev_std)  
az.plot_posterior({'X':X})
plt.show()