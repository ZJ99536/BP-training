import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Normalization

reconstructed_model = keras.models.load_model('model.h5')
dataset = loadtxt('/home/zhoujin/rpg_time_optimal/my_time_optimal/gate.txt', delimiter=',')
# split into input (X) and output (y) variables
input = dataset[110:120,0:11]
output = dataset[110:120,11:17]
# normalizer = Normalization(axis=-1)
# normalizer.adapt(dataset)
# dataset = normalizer(dataset)
print(reconstructed_model.predict(input))
print(output)

