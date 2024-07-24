import numpy as np
from scipy.stats import binom 
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
num = 7072
prob = 0.45
output = []
i = 3000
while i in range(3400):
    output.append(i)
    i += 1
binomial_sample = binom.pmf(output,n = num,p = prob)
poisson_sample = poisson.pmf(output, mu = 3182.59)
normal_sample = norm.pdf(output,3182.59, 41.84)
plt.plot(output, binomial_sample, color = 'r', label = 'binomial')
plt.plot(output, poisson_sample, color = 'b', label = 'poisson')
plt.plot(output, normal_sample, color = 'y', label = 'normal' ,linestyle = 'dashed')
plt.legend()
plt.show()