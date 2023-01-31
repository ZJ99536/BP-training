import numpy as np
from numpy import loadtxt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Dense, Normalization
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# model = keras.models.load_model('/home/zhoujin/learning/model/model1.h5')
dataset = loadtxt('/home/zhoujin/trajectory-generation/trajectory/gatedd.txt', delimiter=',')
# split into input (X) and output (y) variables
