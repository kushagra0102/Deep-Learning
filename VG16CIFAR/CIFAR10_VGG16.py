import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, optimizers, regularizers, callbacks


##Importing the cifar10 Dataset##

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.cifar10.load_data()
classes = np.unique(train_Y)
num_classes = len(classes)

##Reshaping the data. ##
train_X = train_X.reshape(-1, 32,32,3)
test_X= test_X.reshape (-1,32,32,3)

### Normalizing the data.##
X_train = train_X / 255.
X_test = test_X / 255.
X_train = train_X.astype('float32')
X_test = test_X.astype('float32')

# Normalizing the input according to the methods used in the paper. ##
X_train = preprocess_input(X_train)
y_test = to_categorical(test_Y)

# Using one-hot-encode the labels for training
X_test = preprocess_input(X_test)
y_train = to_categorical(train_Y)


# Define the parameters for instanitaing VGG16 model.

vgg16 = VGG16(
    weights=None,
    include_top=False,
    classes = num_classes,
    input_shape=(32,32,3)
)

## Using the SGD Optimizer ##
##Intializing the optimizer##

sgd = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=False,clipvalue=0.5)
l2_reg = regularizers.l2(5e-4)

model = Sequential()
model.add(vgg16)
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2_reg))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2_reg))
model.add(layers.Dropout(0.5))

## Adding extra dense layer##

model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2_reg))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

##Viewing the model summary##

model.summary()

##Compiling the model ##
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

##Creating checkpoint to save the model##

checkpoint = ModelCheckpoint(
    'modelVGG16_CIFAR10.h5',
    monitor='val_acc',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

# Training the model
history = model.fit(
    x=X_train,
    y=y_train,
    validation_split=0.1,
    batch_size=32,
    epochs=100,
    callbacks=[checkpoint],
    verbose=1
)

## Fetching history details from the trained model##

accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
Train_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

##Plotting the Accuracy for the Model##

plt.title('Accuracy Plot')
plt.plot(epochs, accuracy, 'green', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, 'brown', label='Validation Accuracy')
legend = plt.legend(loc='upper center', shadow=True, fontsize='small')
legend.get_frame().set_facecolor('C0')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.grid(True)
plt.savefig('/content/sample_data/Accuracy.png')
plt.show()


##Plotting the Loss for the Model##
plt.title('Loss Plot')
plt.plot(epochs, Train_loss, 'green', label='Training loss')
plt.plot(epochs, validation_loss, 'brown', label='Validation loss')
legend = plt.legend(loc='best', shadow=True, fontsize='small')
legend.get_frame().set_facecolor('C0')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.grid(True)
plt.savefig('/content/sample_data/image/Loss.png')
plt.show()


## Printing the test loss and test accuracy for the model##
test_loss, test_score = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_score)
