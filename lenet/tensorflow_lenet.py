from tensorflow.keras.layers import Dense,Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential

model = Sequential([
    Conv2D(filters=6,kernel_size=5,strides=1,activation="relu"),
    MaxPooling2D(pool_size=2,strides=1),
    Conv2D(filters=16,kernel_size=5,strides=1,activation="relu"),
    MaxPooling2D(pool_size=2,strides=2),
    Conv2D(filters=120,kernel_size=5,strides=1),
    Flatten(),
    Dense(units=84,activation="relu"),
    Dense(units=10,activation="softmax")
])

model.build(input_shape=(None, 28, 28, 1))  #just for testing purpose
model.summary()