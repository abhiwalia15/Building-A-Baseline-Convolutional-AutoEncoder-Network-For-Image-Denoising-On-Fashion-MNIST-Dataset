from keras.models import Model
from imgaug import augmenters
from random import randint
import pandas as pd
import numpy as np
from numpy import argmax, array_equal
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import MaxPool2D
from keras.layers import UpSampling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import fashion_mnist

#Load the data from keras
(X_train, y), (X_test, y) = fashion_mnist.load_data()
#Undo their train and test so we can manually set out own split
X = np.vstack([X_train, X_test])

train_X, val_X = train_test_split(X, test_size=0.2)

#Normalize the images to (0, 1)
train_X = train_X/255.
val_X = val_X/255.

#Reshape to (num_samples, height, width, depth)
train_X = train_X.reshape(-1, 28, 28, 1)
val_X = val_X.reshape(-1, 28, 28, 1)

#Add noise
Noise = augmenters.SaltAndPepper(0.1)
Seq_Obj = augmenters.Sequential([Noise])

train_X_Noise = Seq_Obj.augment_images(train_X * 255) / 255
val_X_Noise = Seq_Obj.augment_images(val_X * 255) / 255

#Define the model
Input_layer = Input(shape=(28, 28, 1))

Encoded_layer_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(Input_layer)
Encoded_layer_1 = MaxPool2D( (2, 2), padding='same')(Encoded_layer_1)

Encoded_layer_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(Encoded_layer_1)
Encoded_layer_2 = MaxPool2D( (2, 2), padding='same')(Encoded_layer_2)

Encoded_layer_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(Encoded_layer_2)

Latent_view    = MaxPool2D( (2, 2), padding='same')(Encoded_layer_3)

Decoded_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(Latent_view)
Decoded_layer_1 = UpSampling2D((2, 2))(Decoded_layer_1)

Decoded_layer_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(Decoded_layer_1)
Decoded_layer_2 = UpSampling2D((2, 2))(Decoded_layer_2)

Decoded_layer_3 = Conv2D(64, (3, 3), activation='relu')(Decoded_layer_2)
Decoded_layer_3 = UpSampling2D((2, 2))(Decoded_layer_3)

Output_layer   = Conv2D(1, (3, 3), padding='same')(Decoded_layer_3)

Fashion_Model = Model(Input_layer, Output_layer)

Fashion_Model.compile(optimizer='adam', loss='mse')

Fashion_Model.summary()

#Define the checkpoint so the model saves the best version
model_checkpoint_callback = ModelCheckpoint(filepath = "autoencoder_best_val.h5", monitor = 'val_loss', save_best_only = True)

#Fit with 1000 epochs, 2048 batch size
History = Fashion_Model.fit(train_X_Noise, train_X, epochs=1000, batch_size=2048, validation_data=(val_X_Noise, val_X), callbacks=[model_checkpoint_callback])

#Predict on the first ten samples of our validation data
predictions = Fashion_Model.predict(val_X_Noise[:10])

#Graph the results, save the loss
fsh, aix = plt.subplots(1,5)

for i in range(5,10):
    aix[i-5].imshow(predictions[i].reshape(28, 28))
plt.show()

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('loss')






























