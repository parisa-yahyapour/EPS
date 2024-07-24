import numpy as np, random
import matplotlib.pyplot as plt
import scipy.stats as stats 
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(810109203)

sample1 = np.random.poisson(3,5000)
plt.title("poisson 3")
plt.hist(sample1, color='b', density=True)
plt.show()

standart_deviation1 = np.std(np.array(sample1))
mean_sample1 = np.mean(np.array(sample1))
print("standard deviation1 = ", standart_deviation1)
print("mean1 = ", mean_sample1)

observation1 = np.random.normal(mean_sample1, standart_deviation1, 5000)
z1 = (observation1-np.mean(observation1))/np.std(observation1) 
  
stats.probplot(z1, dist="norm", plot=plt) 
plt.title("Q-Q plot 5000 = n") 
plt.show()

statistic1, p_value1 = stats.shapiro(sample1)
print("p value1 = ", p_value1) 

sample2 = np.random.poisson(3,50)

standart_deviation2 = np.std(np.array(sample2))
mean_sample2 = np.mean(np.array(sample2))
print("standard deviation2 = ", standart_deviation2)
print("mean2 = ", mean_sample2)

observation2 = np.random.normal(mean_sample2, standart_deviation2, 50)
z2 = (observation2-np.mean(observation2))/np.std(observation2) 
stats.probplot(z2, dist="norm", plot=plt) 
plt.title("Q-Q plot 50 = n") 
plt.show()

statistic2, p_value2 = stats.shapiro(sample2)
print("p value2 = ", p_value2) 

sample3 = np.random.poisson(3,5)

standart_deviation3 = np.std(np.array(sample3))
mean_sample3 = np.mean(np.array(sample3))
print("standard deviation3 = ", standart_deviation3)
print("mean3 = ", mean_sample3)

observation3 = np.random.normal(mean_sample3, standart_deviation3, 5)
z3 = (observation3-np.mean(observation3))/np.std(observation3) 
  
stats.probplot(z3, dist="norm", plot=plt) 
plt.title("Q-Q plot 5 = n") 
plt.show()

statistic3, p_value3 = stats.shapiro(sample3)
print("p value3 = ", p_value3) 