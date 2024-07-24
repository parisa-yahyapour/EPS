import numpy as np
from scipy.stats import poisson
from scipy.stats import expon
from scipy.stats import uniform
from matplotlib import pyplot as plt
from matplotlib import colors
poisson_sample = np.random.poisson(14,1000)
exponentioal_sample = np.random.exponential(6,1000)
unifrom_sample = np.random.uniform(0,21,1000)
temp = poisson_sample + exponentioal_sample
sum_sample = temp + unifrom_sample
plt.hist(poisson_sample, color='r', alpha=0.3, label='DM')
plt.hist(exponentioal_sample, color='y',alpha=0.3, label='AP')
plt.hist(unifrom_sample, color='b',alpha=0.3, label='PY')
plt.hist(sum_sample, color='m',alpha=0.3, label='sum')
plt.legend()
plt.show()