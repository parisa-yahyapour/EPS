import numpy as np
from scipy.stats import binom 
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
num = 250
prob = 0.008
output = list(range(5))
binomial_sample = [binom.pmf(r,n = num,p = prob) for r in output]
poisson_sample = [poisson.pmf(r, mu = 2) for r in output]
normal_sample = [norm.pdf(r,2, 1.984) for r in output]
plt.plot(output, binomial_sample, color = 'r', label = 'binomial')
plt.plot(output, poisson_sample, color = 'b', linestyle = 'dashed', label = 'poisson')
plt.plot(output, normal_sample, color = 'g', label = 'normal')
plt.legend()
plt.show()