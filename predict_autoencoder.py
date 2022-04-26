import tensorflow as tf
import keras
from keras.datasets import fashion_mnist

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Conv2D, BatchNormalization, UpSampling2D

import matplotlib.pyplot as plt

import numpy as np
import pickle
import cv2
import os

#Autoencoder architecture definition
def create_model():
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

    print(Fashion_Model.summary())
    
    return Fashion_Model

model = create_model()

#Load the weights into the model and the testing dataset
model.load_weights("autoencoder_best_val.h5")
metadata = pickle.load(open("metadata.pickle", "rb"))

#Make predictions and compare to the noisy images for all images in the testing set
os.makedirs("predictions/", exist_ok = True)

fig, axes = plt.subplots(1, 2)
for i, image in enumerate(metadata['test']):
    image = np.expand_dims(image, axis = 0)
    image = np.expand_dims(image, axis = -1) # (1, 28, 28, 1)
    prediction = model.predict(image)
    prediction = np.reshape(prediction, (28, 28))
    
    axes[0].imshow(image[0, :, :, 0])
    axes[1].imshow(prediction)

    plt.savefig("predictions/Sample %d.png" % i)
    
    axes[0].cla()
    axes[1].cla()
    
