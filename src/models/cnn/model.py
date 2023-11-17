from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, InputLayer
from tensorflow.python.keras.models import Sequential


def create_model(input_shape):
    # Create model:
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    return model
