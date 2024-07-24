import pandas as pd
from matplotlib import pyplot as plt
#1
data = pd.read_csv("digits.csv")
index_201 = data.iloc[:,202]
index_202 = data.iloc[:,203]
data.drop(data.columns[[202,203]], axis=1, inplace=True)
#2
threshold = 128
for i in range(1,len(data.axes[1])):
    data.iloc[:, i] = data.iloc[:, i].apply(lambda u : 0 if u <= threshold else 1)




