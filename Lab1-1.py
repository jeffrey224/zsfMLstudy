import tensorflow as tf
import numpy as np
from tensorflow import keras

# model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8,input_shape=[2]),
    # tf.keras.layers.Dense(512),
    # tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(512, activation='tanh'),
    tf.keras.layers.Dense(1024, activation='tanh'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(512, activation='tanh'),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1)
])

# model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

xs = np.array([(-1.0,1.0),(0.0, 1.0), (2.0, 3.0), (4.0,3.0),(5.0,6.0),(-2.0,2.0),(3.0,3.0),(3.2,3.4),(7.0,2.0),(5.0,5.0),(-5.0,4.0)], dtype=float)
ys = np.array([2.0, 1.0, 7.0, 19.0, 31.0, 6.0,12.0,13.64,51.0,30.0,29.0], dtype=float)

model.fit(xs, ys, epochs=1000)
print(model.predict([(-5.0,3.0), (2.0,3.0),(5.0,9.0)]))
