from keras.models import Sequential
from keras.layers import Dense, Activation
import keras

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model = Sequential()
# Specifying the input shape
# first layer needs to know what input shape to expect other layers do it automatically
    # input shape - shape tuple (tuple of integers), batch dimension not included
    # input_dim - for 2D layers, or 3D layers with input_length
    # batch_size - fixed batch szie (useful for stateful recurrent networks)
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

# Compilation
# compile method = learning process | arguments:
    # optimizer - how models will train to minimize required time without penalizing the cost
    # loss function - method of evaluating how well specific algorithm models the given data
    # list of metrics

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

# Training
# input data = Numpy arrays | fit method = training

# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)