# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
from tensorflow import keras
# load the dataset
dataset = loadtxt('/home/zhoujin/trajectory-generation/trajectory/gate.txt', delimiter=',')
# split into input (X) and output (y) variables

normalizer = Normalization(axis=-1, invert=True)
normalizer.adapt(dataset)
dataset = normalizer(dataset)
X = dataset[:,0:11]
y = dataset[:,11:17]


# define the keras model
model = Sequential()
model.add(Dense(32, input_shape=(11,), activation='relu'))
# model.add(Dense(32, input_shape=(11,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
# model.add(Dense(128, activation='sigmoid'))
model.add(Dense(6, activation='softmax'))
model.summary()
# compile the keras model
# model.compile(loss = "categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.compile(loss = "mse", optimizer="adam", metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=60, batch_size=5, validation_split=0.2)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
model.save('/home/zhoujin/learning/model/model.h5')