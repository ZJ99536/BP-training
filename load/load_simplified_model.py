import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Normalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

model = keras.models.load_model('/home/zhoujin/learning/model/model.h5')
dataset = loadtxt('/home/zhoujin/trajectory-generation/trajectory/gated.txt', delimiter=',')
# split into input (X) and output (y) variables

X = dataset[:,0:8]
y = dataset[:,8:10]

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X)
X = min_max_scaler.transform(X)

min_max_scaler.fit(y)
remember = min_max_scaler
y = min_max_scaler.transform(y)

input = X[0:32,0:8]
output = dataset[0:32,8:10]

ynew = model.predict(input)
ynew = remember.inverse_transform(ynew)

plt.plot(ynew)
plt.plot(output)
plt.show()
# print(remember.inverse_transform(ynew))
# print(output)

