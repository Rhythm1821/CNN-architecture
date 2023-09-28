from tensorflow.keras.layers import Dense,Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras import Sequential

model = Sequential([
    Conv2D(96,kernel_size=11,strides=4,activation="relu",input_shape=(227,227,3)),
    MaxPooling2D(pool_size=3,strides=2),
    Conv2D(256,kernel_size=5,strides=1,padding="same",activation="relu"),
    MaxPooling2D(pool_size=3,strides=2),
    Conv2D(384,kernel_size=3,strides=1,activation="relu"),
    Conv2D(384,kernel_size=3,strides=1,activation="relu"),
    Conv2D(256,kernel_size=3,strides=1,activation="relu"),
    MaxPooling2D(pool_size=3,strides=2),
    Flatten(),
    Dense(4096,activation="relu"),
    Dropout(0.5),
    Dense(4096,activation="relu"),
    Dropout(0.5),
    Dense(10,activation="softmax"),
])

model.build(input_shape=(None,227,227,3))
model.summary()