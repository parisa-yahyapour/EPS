import numpy as np, random
import matplotlib.pyplot as plt
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(810109203)

import pandas as pd
from keras.datasets import mnist
(_,_), (test_images, _) = mnist.load_data()
test_images = test_images.reshape(test_images.shape[0], -1)
test_images = test_images.astype('float32') / 255.0

import tensorflow as tf
autoencoder = tf.keras.models.load_model('mnist_AE.h5')
reconstructed_images = autoencoder.predict(test_images)
for i in range(4):
  num = np.random.randint(10000)
  temp1 = test_images[num].reshape(28,28)
  temp2 = reconstructed_images[num].reshape(28,28)
  plt.imshow(temp1)
  plt.show()
  plt.imshow(temp2)
  plt.show()

def MSE_calculator(main , reconstructed):
   value = np.mean((main - reconstructed)**2, axis=1)
   return value
mse = MSE_calculator(test_images, reconstructed_images)
plt.hist(mse, density=True, bins= 50, color='r')
plt.title("MSE")
plt.show()

standard_deviation = np.std(mse)
mean_sampe = np.mean(mse)
from scipy import stats
ks_statistics, p_value = stats.kstest(mse, cdf='norm', args=(mean_sampe, standard_deviation))
print("p_value= ",p_value)
