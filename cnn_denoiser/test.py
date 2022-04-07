import tensorflow as tf
import keras
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Conv2D


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train[0])

def create_model():
    input = Input(shape = (28, 28, 1))
        
    x = Conv2D(16, (3, 3), strides = (1, 1), padding = "same", activation = "relu")(input)
        
    x = Flatten()(x)

    x = Dense(10, activation = "softmax")(x)

    model = Model(inputs = input, outputs = x)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])
    
    return model

model = create_model()

history = model.fit(X_train, y_train, batch_size = 1024, epochs = 10)

print(model.evaluate(X_test, y_test))
