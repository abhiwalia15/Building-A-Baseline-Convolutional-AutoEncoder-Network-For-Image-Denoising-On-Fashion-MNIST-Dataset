import tensorflow as tf
import keras
from keras.datasets import fashion_mnist

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Conv2D, BatchNormalization

from imgaug import augmenters
import matplotlib.pyplot as plt

import numpy as np
import pickle


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

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = "model_best_val.h5",
    monitor = 'val_loss',
    save_best_only = True)
    
model = create_model()


(X_train, y), (X_test, y) = fashion_mnist.load_data()

y = np.vstack([X_train, X_test]) / 255

np.random.shuffle(y)
y_train = y[0: 50000]
y_val = y[50000: 60000]
y_test = y[60000:]

print("Augmenting...")
noise = augmenters.SaltAndPepper(0.1)
seq_object = augmenters.Sequential([noise])

X_train = seq_object.augment_images(y_train * 255) / 255
X_val = seq_object.augment_images(y_val * 255) / 255
X_test = seq_object.augment_images(y_test * 255) / 255
print("Augmentation complete")

y_train = np.reshape(y_train, (y_train.shape[0], 784))
y_val = np.reshape(y_val, (y_val.shape[0], 784))
y_test = np.reshape(y_test, (y_test.shape[0], 784))

metadata = {'train': X_train, 'validation': X_val, 'test': X_test}
pickle.dump(metadata, open("metadata.pickle", "wb"))

history = model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = 1000, epochs = 600, callbacks = [model_checkpoint_callback])

plt.plot(history.history['loss'], color = "blue", label = "Training Loss (min " + str(round(np.array(history.history['loss']).min(), 7)) + ")")
plt.plot(history.history['val_loss'], color = "red", label = "Validation Loss (min " + str(round(np.array(history.history['val_loss']).min(), 7)) + ")")

handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(handles, labels)

plt.savefig("loss_curves.png")

plt.clf()

pickle.dump(history.history, open("history.pickle", "wb"))