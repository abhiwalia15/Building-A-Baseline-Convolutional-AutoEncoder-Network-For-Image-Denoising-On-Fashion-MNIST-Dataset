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

#CNN architecture definition
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

#Load the weights into the model and the testing dataset
model.load_weights("model_best_val.h5")
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
    
