import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

train_x = None
train_y = None

with open('./train_x.pkl', "rb") as f:
    train_x = pickle.Unpickler(f).load()
with open('./train_y.pkl', "rb") as f:
    train_y = pickle.Unpickler(f).load()

model = Sequential()
model.add(Dense(160, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('./model-data', include_optimizer=True, save_format='hdf5')

