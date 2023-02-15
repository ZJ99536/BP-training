import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Normalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

model = keras.models.load_model('/home/zhoujin/learning/model/quad4_val.h5')
# dataset = loadtxt('/home/zhoujin/trajectory-generation/trajectory/quad2.txt', delimiter=',')
dataset = loadtxt('/home/zhoujin/cloud-communiacation/library/quad1.txt', delimiter=',')
# split into input (X) and output (y) variables

X = dataset[:,0:6]
y = dataset[:,34:36]

# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(X)
# X = min_max_scaler.transform(X)

# min_max_scaler.fit(y)
# remember = min_max_scaler
# y = min_max_scaler.transform(y)

input = X[611833:611994,0:6]
output = dataset[611833:611994,34:36]
# output_origin = y[61833:61894,:]

ynew = model(input)
# ynew = remember.inverse_transform(ynew)

# print(ynew)
# print(output)

plt.plot(ynew[:,0:2])
# plt.plot(ynew[:,16:18])
# plt.plot(output_origin[:,6:7])
plt.plot(output[:,0:2])
# plt.plot(output[:,16:18])
plt.show()
# print(remember.inverse_transform(ynew))
# print(output)

