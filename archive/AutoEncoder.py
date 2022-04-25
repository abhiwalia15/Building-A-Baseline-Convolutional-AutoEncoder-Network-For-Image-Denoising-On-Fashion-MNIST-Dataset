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


df = pd.read_csv("fashion-mnist_train.csv")

train_X = df[list(df.columns)[1:]].values
train_X, val_X = train_test_split(train_X, test_size=0.2)

train_X = train_X/255.
val_X = val_X/255.

train_X = train_X.reshape(-1, 28, 28, 1)
val_X = val_X.reshape(-1, 28, 28, 1)

Noise = augmenters.SaltAndPepper(0.1)
Seq_Obj = augmenters.Sequential([Noise])

train_X_Noise = Seq_Obj.augment_images(train_X * 255) / 255
val_X_Noise = Seq_Obj.augment_images(val_X * 255) / 255

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

Fashion_Model.compile(optimizer='adam', loss='mse', metrics=["acc"])

#Fashion_Model.summary()

cp_EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=5, mode='auto')

model_checkpoint_callback = ModelCheckpoint(filepath = "model_best_val.h5", monitor = 'val_loss', save_best_only = True)

History = Fashion_Model.fit(train_X_Noise, train_X, epochs=1000, batch_size=2048, validation_data=(val_X_Noise, val_X_Noise), callbacks=[cp_EarlyStopping, model_checkpoint_callback])

predictions = Fashion_Model.predict(val_X_Noise[:10])

fsh, aix = plt.subplots(1,5)

fsh.set_size_inches(90, 50)

for i in range(5,10):
    aix[i-5].imshow(predictions[i].reshape(28, 28))
plt.show()

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.savefig('loss')

plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.savefig('accuracy')






























