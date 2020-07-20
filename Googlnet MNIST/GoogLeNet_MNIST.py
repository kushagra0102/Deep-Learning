import tensorflow as tf
import collections
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import keras
from keras.layers import Input, Dense, Conv2D, MaxPool2D, GlobalAvgPool2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical

##Defing the parameters for the model##
epochs = 100
batch_size = 128

##Loading the MNIST dataset##

mnist = keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()

##Stacking the layers to make it from 1 to 3 channels##

x_train=np.dstack([x_train] * 3)
x_test=np.dstack([x_test]*3)

##Reshaping the data to three channels##

x_train=x_train.reshape(-1,28,28,3)
x_test=x_test.reshape(-1,28,28,3)

# We one-hot-encode the labels for training

y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=None, dtype='float32')
y_test = to_categorical(y_test)

##Defning the optimizer for the model##
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-3)
objective = 'binary_crossentropy'

##Creating the model##
def googlenet():
  def inception_block(x,f):
    t1=Conv2D(f[0],1,activation='relu')(x)
    t2= Conv2D(f[1],1,activation='relu')(x)
    t2= Conv2D(f[2],3,padding='same', activation='relu')(t2)

    t3= Conv2D(f[3],1,activation='relu')(x)
    t3= Conv2D(f[4],5,padding='same', activation='relu')(t3)

    t4=  MaxPool2D(3, strides=1,padding='same')(x)
    t4= Conv2D(f[5],1, activation='relu')(t4)

    output= Concatenate()([t1,t2,t3,t4])
    return output

  input=Input(shape=(28,28,3))
  x= Conv2D(64,7,strides=2, padding='same', activation='relu')(input)
  x= MaxPool2D(3,strides=2,padding='same')(x)
  x=Conv2D(64,1,activation='relu')(x)
  x= Conv2D(192,3, padding='same',activation='relu')(x)
  x= MaxPool2D(3, strides=2, padding='same')(x)

  x=inception_block(x,[64,96,128,16,32,32])
  x=inception_block(x,[128,128,192,32,96,64])
  x= MaxPool2D(3, strides=2, padding='same')(x)

  x=inception_block(x,[192,96,208,16,48,64])
  x=inception_block(x,[160,112,224,24,64,64])
  x=inception_block(x,[128,128,256,24,64,64])
  x=inception_block(x,[112,144,288,32,64,64])
  x=inception_block(x,[256,160,320,32,128,128])
  x= MaxPool2D(3, strides=2, padding='same')(x)

  x=inception_block(x,[256,160,320,32,128,128])
  x=inception_block(x,[384,192,384,48,128,128])
  x=GlobalAvgPool2D()(x)
  x=Dropout(0.4)(x)

  output=Dense(10,activation='softmax')(x)
  model=Model(input,output)
  return model

model=googlenet()


##Viewing the model summary##
model.summary()

##Compiling the model ##
model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

##Early stopping definition##

early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')

##Training the model##

history = model.fit(x_train, y_train_onehot, batch_size=batch_size, epochs=epochs,
              validation_split=0.25, verbose=1, shuffle=True, callbacks=[early_stopping])

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
legend = plt.legend(loc='best', shadow=True, fontsize='small')
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
plt.savefig('/content/sample_data/Loss.png')
plt.show()

## Printing the test loss and test accuracy for the model##
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
