# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# load the dataset
dataset = loadtxt('/home/zhoujin/trajectory-generation/trajectory/quad1.txt', delimiter=',')
# split into input (X) and output (y) variables

# normalizer = Normalization(axis=-1, invert=True)
# normalizer.adapt(dataset)
# dataset = normalizer(dataset)
X = dataset[:,0:6]
y = dataset[:,8:14]

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X)
X = min_max_scaler.transform(X)

min_max_scaler.fit(y)
remember = min_max_scaler
y = min_max_scaler.transform(y)

# define the keras model
model = Sequential()
model.add(Dense(10, input_shape=(6,), activation='relu'))
# model.add(Dense(32, input_shape=(11,), activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(128, activation='sigmoid'))
model.add(Dense(6, activation='linear'))
model.summary()
# compile the keras model
# model.compile(loss = "categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.compile(loss = "mse", optimizer="adam", metrics=['accuracy'])
# fit the keras model on the dataset
history = model.fit(X, y, epochs=2, batch_size=3, validation_split=0.2)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

model.save('/home/zhoujin/learning/model/quad1_fast.h5')

plt.plot(history.history['loss'])
plt.show()

# input = X[:,0:8]
# output = dataset[:,8:10]

# ynew = model.predict(input)
# ynew = remember.inverse_transform(ynew)

# plt.plot(ynew)
# plt.plot(output)
# plt.show()