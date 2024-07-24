import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def coefficient_calculator(x,y):
    num_observation = np.size(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    xy = np.sum(y*x) - num_observation*mean_y*mean_x
    xx = np.sum(x*x) - num_observation*mean_x*mean_x
    regression_coefficient1 = xy / xx
    regression_coefficient2 = mean_y - regression_coefficient1*mean_x
    return (regression_coefficient2, regression_coefficient1)

def regression_shape(x , y, regression_coefficient):
    plt.scatter(x , y , color='m',marker='o', s=30 )
    prediction_y = regression_coefficient[0] + regression_coefficient[1]*x
    print("r^2 = ", r2_score(y, prediction_y))
    plt.plot(x, prediction_y, color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plt.title("main 8 data")
x_regression1 = [-2.3, -1.1, 0.5, 3.2, 4.0, 6.7, 10.3, 11.5]
y_regression1 = [-9.6, -4.9, -4.1, 2.7, 5.9, 10.8, 18.9, 20.5]
coefficient1 = coefficient_calculator(np.array(x_regression1), np.array(y_regression1))
regression_shape(np.array(x_regression1), np.array(y_regression1), coefficient1)

plt.title("main 8 data with outlier")
x_regression2 = [-2.3, -1.1, 0.5, 3.2, 4.0, 6.7, 10.3, 11.5, 5.8]
y_regression2 = [-9.6, -4.9, -4.1, 2.7, 5.9, 10.8, 18.9, 20.5, 31.3]
coefficient2 = coefficient_calculator(np.array(x_regression2), np.array(y_regression2))
regression_shape(np.array(x_regression2), np.array(y_regression2), coefficient2)

plt.title("main 8 data with high leverage point")
x_regression3 = [-2.3, -1.1, 0.5, 3.2, 4.0, 6.7, 10.3, 11.5, 20.4]
y_regression3 = [-9.6, -4.9, -4.1, 2.7, 5.9, 10.8, 18.9, 20.5, 14.1]
coefficient3 = coefficient_calculator(np.array(x_regression3), np.array(y_regression3))
regression_shape(np.array(x_regression3), np.array(y_regression3), coefficient3)

plt.title("main 8 data with outlier-high leverage point")
x_regression4 = [-2.3, -1.1, 0.5, 3.2, 4.0, 6.7, 10.3, 11.5, 20.4]
y_regression4 = [-9.6, -4.9, -4.1, 2.7, 5.9, 10.8, 18.9, 20.5, 31.3]
coefficient4 = coefficient_calculator(np.array(x_regression4), np.array(y_regression4))
regression_shape(np.array(x_regression4), np.array(y_regression4), coefficient4)