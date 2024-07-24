import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.stats import poisson
from scipy.stats import binom

tarbiat = pd.read_csv("Tarbiat.csv")
ls_metro = tarbiat["metro"].tolist()
ls_BRT = tarbiat["BRT"].tolist()
plt.hist(ls_metro, color='r', alpha=0.3, label='Metro')
plt.hist(ls_BRT, color='b', alpha=0.3, label='BRT')
plt.legend()
plt.show()
p = tarbiat.mean(axis=0)
print("paramters of poisson distribution: \n", "metro: ", p[0], " BRT: ", p[1])
plt.hist(ls_metro,bins=range(int(min(ls_metro)), int(max(ls_metro))+2), color='m', label='Metro_density', density=True)
x = np.arange(int(min(ls_metro)), int(max(ls_metro))+2)
y = poisson.pmf(x, p[0])
plt.plot(x, y, color = 'b', label="Poisson")
plt.legend()
plt.show()
parameter_z = p[0] + p[1]
print("parameter z = ", parameter_z)
numpy_metro = np.array(ls_metro)
numpy_BRT = np.array(ls_BRT)
z = numpy_BRT + numpy_metro
plt.hist(z, bins=range(np.min(z), np.max(z) + 2), color = 'y', label="X+Y", density=True)
h_z = np.arange(np.min(z), np.max(z) + 2)
v_z = poisson.pmf(h_z, parameter_z)
plt.plot(h_z, v_z, color = 'r', label="Poisson Z")
plt.legend()
plt.show()
w_parameter = p[0]/parameter_z
print("parameter of W: ", w_parameter)
w = binom.pmf(x , 8 , w_parameter)
plt.plot(x,w,color="blue", label='X|8')
plt.legend()
plt.show()
part8 = []
for i in range(len(ls_metro)):
    if ls_metro[i] + ls_BRT[i] == 8:
        part8.append(ls_metro[i])
plt.hist(part8, bins=range(0,14), color = 'r', label="w based on data", density=True)
plt.legend()
plt.show()
