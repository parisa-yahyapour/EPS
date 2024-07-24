import numpy as np
import matplotlib.pyplot as plt
def generate_binomial(num_binomial : int, num_bernouili : int, success_prob : float)-> np.array:
    prob = [1 - success_prob , success_prob]
    berouili_samples = np.random.choice(2,num_bernouili*num_binomial,p = prob)
    binomial_sample = np.resize(berouili_samples,(num_binomial,num_bernouili))
    return np.sum(binomial_sample,axis=1)
p = []
i = 0
for i in range(101):
    p.append(i/100)
avg_theory = []
var_theory = []
avg = []
var = []
n = 500
m = 5000
for i in range(len(p)):
    binomial = generate_binomial(m,n,p[i])
    avg_theory.append(n * p[i])
    var_theory.append(n * p[i] * (1 - p[i]))
    avg.append(np.mean(binomial))
    var.append(np.var(binomial))
plt.plot(p,avg_theory, color = 'c', label = 'theory_average')
plt.plot(p,avg, color = 'g',  label = 'real_average', linestyle = 'dashed')
plt.legend()
plt.show()
plt.plot(p,var_theory, color = 'r',  label = 'theory_variance')
plt.plot(p,var, color = 'y', label = 'real_variance')
plt.legend()
plt.show()