import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Normalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

model = keras.models.load_model('/home/zhoujin/learning/model/model1.h5')
dataset = loadtxt('/home/zhoujin/trajectory-generation/trajectory/gatedd.txt', delimiter=',')
# split into input (X) and output (y) variables

X = dataset[:,0:8]
y = dataset[:,8:10]

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X)
X = min_max_scaler.transform(X)

min_max_scaler.fit(y)
remember = min_max_scaler
y = min_max_scaler.transform(y)

input = X[6110:6211,0:8]
output = dataset[6110:6211,8:10]

ynew = model.predict(input)
ynew = remember.inverse_transform(ynew)

print(ynew)
print(output)

plt.plot(ynew[:,0:1])
plt.plot(output[:,0:1])
plt.show()
print(remember.inverse_transform(ynew))
print(output)

