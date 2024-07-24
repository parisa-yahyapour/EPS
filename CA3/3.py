import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 

import numpy as np, random
import matplotlib.pyplot as plt
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(810109203)

df = pd.read_csv('FIFA2020.csv', encoding="ISO-8859-1")
mean_pace = df["pace"].mean()
mean_dribbling = df["dribbling"].mean()
df.loc[df["pace"].isnull() , "pace"] = mean_pace
df.loc[df["dribbling"].isnull() , "dribbling"] = mean_dribbling

fig = plt.figure(figsize=(10,10))
plt.boxplot(df["age"])
plt.show()

min_age = np.min(df["age"])
print("min = ",min_age)

max_age = np.max(df["age"])
print("max = ", max_age)

q1 = np.percentile(df["age"], 25)
print("Q1 = ", q1)

q2 = np.percentile(df["age"], 50)
print("Q2 = ", q2)

q3 = np.percentile(df["age"], 75)
print("Q3 = ", q3)

index_weight = df.columns.get_loc("weight")
random_sample = []
for i in range(100):
    num = np.random.randint(len(df))
    random_sample.append(df.iloc[i,index_weight])
    
variance = np.var(np.array(random_sample))
standart_deviation = np.std(np.array(random_sample))
mean_sample = np.mean(np.array(random_sample))
print("variance = ", variance)
print("standard deviation = ", standart_deviation)
print("mean = ", mean_sample)

observation = np.random.normal(mean_sample, standart_deviation, 100)
z = (observation-np.mean(observation))/np.std(observation) 
  
stats.probplot(z, dist="norm", plot=plt) 
plt.title("Q-Q plot") 
plt.show()

statistic, p_value = stats.shapiro(random_sample)
print("p value = ", p_value) 


random_sample2 = []
for i in range(500):
    num = np.random.randint(len(df))
    random_sample2.append(df.iloc[i,index_weight])
    
variance2 = np.var(np.array(random_sample2))
standart_deviation2 = np.std(np.array(random_sample2))
mean_sample2 = np.mean(np.array(random_sample2))
print("variance2 = ", variance2)
print("standard deviation2 = ", standart_deviation2)
print("mean2 = ", mean_sample2)

observation2 = np.random.normal(mean_sample2, standart_deviation2, 500)
z2 = (observation2-np.mean(observation2))/np.std(observation2) 
  
stats.probplot(z2, dist="norm", plot=plt) 
plt.title("Q-Q plot2") 
plt.show()

statistic2, p_value2 = stats.shapiro(random_sample2)
print("p value2 = ", p_value2) 

random_sample3 = []
for i in range(2000):
    num = np.random.randint(len(df))
    random_sample3.append(df.iloc[i,index_weight])
    
variance3 = np.var(np.array(random_sample3))
standart_deviation3 = np.std(np.array(random_sample3))
mean_sample3 = np.mean(np.array(random_sample3))
print("variance3 = ", variance3)
print("standard deviation3 = ", standart_deviation3)
print("mean3 = ", mean_sample3)

observation3 = np.random.normal(mean_sample3, standart_deviation3, 2000)
z3 = (observation3-np.mean(observation3))/np.std(observation3) 
  
stats.probplot(z3, dist="norm", plot=plt) 
plt.title("Q-Q plot3") 
plt.show()

statistic3, p_value3 = stats.shapiro(random_sample3)
print("p value3 = ", p_value3) 

