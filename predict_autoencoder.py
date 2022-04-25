import tensorflow as tf
import keras
from keras.datasets import fashion_mnist

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Conv2D, BatchNormalization

import matplotlib.pyplot as plt

import numpy as np
import pickle
import cv2
import os

def create_model():
    input = Input(shape = (28, 28, 1))
    
    x = Conv2D(32, (3, 3), strides = (1, 1), padding = "valid", activation = "tanh")(input)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Dropout(0.2)(x)
    
    x = Conv2D(64, (3, 3), strides = (1, 1), padding = "valid", activation = "tanh")(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), strides = (1, 1), padding = "valid", activation = "tanh")(x)
    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), strides = (1, 1), padding = "valid", activation = "tanh")(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)

    x = Dropout(0.2)(x)

    x = Dense(328)(x)

    x = Dropout(0.2)(x)
    
    x = Dense(784)(x)

    model = Model(inputs = input, outputs = x)
    model.compile(optimizer = "adam", loss = "mse")
    
    print(model.summary())
    return model

model = create_model()









from keras.layers import UpSampling2D
Input_layer = Input(shape=(28, 28, 1))

Encoded_layer_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(Input_layer)
Encoded_layer_1 = MaxPooling2D( (2, 2), padding='same')(Encoded_layer_1)

Encoded_layer_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(Encoded_layer_1)
Encoded_layer_2 = MaxPooling2D( (2, 2), padding='same')(Encoded_layer_2)

Encoded_layer_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(Encoded_layer_2)

Latent_view    = MaxPooling2D( (2, 2), padding='same')(Encoded_layer_3)

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

Fashion_Model.load_weights("autoencoder_best_val.h5")
metadata = pickle.load(open("metadata.pickle", "rb"))

os.makedirs("predictions/", exist_ok = True)

fig, axes = plt.subplots(1, 2)
for i, image in enumerate(metadata['test']):
    image = np.expand_dims(image, axis = 0)
    image = np.expand_dims(image, axis = -1) # (1, 28, 28, 1)
    prediction = Fashion_Model.predict(image)
    prediction = np.reshape(prediction, (28, 28))
    
    axes[0].imshow(image[0, :, :, 0])
    axes[1].imshow(prediction)

    plt.savefig("predictions/Sample %d.png" % i)
    
    axes[0].cla()
    axes[1].cla()
    
