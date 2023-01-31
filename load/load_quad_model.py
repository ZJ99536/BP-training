import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Normalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

model = keras.models.load_model('/home/zhoujin/learning/model/quad2_m4.h5')
dataset = loadtxt('/home/zhoujin/trajectory-generation/trajectory/quad2.txt', delimiter=',')
# split into input (X) and output (y) variables

X = dataset[:,0:15]
y = dataset[:,18:36]

# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(X)
# X = min_max_scaler.transform(X)

# min_max_scaler.fit(y)
# remember = min_max_scaler
# y = min_max_scaler.transform(y)

input = X[61733:61794,0:15]
output = dataset[61733:61794,18:36]
output_origin = y[61733:61794,:]

ynew = model.predict(input)
# ynew = remember.inverse_transform(ynew)

# print(ynew)
# print(output)

plt.plot(ynew[:,16:18])
# plt.plot(output_origin[:,6:7])
plt.plot(output[:,16:18])
plt.show()
# print(remember.inverse_transform(ynew))
# print(output)

