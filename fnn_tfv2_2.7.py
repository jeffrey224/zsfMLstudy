import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load data
train_data = np.loadtxt("train.txt").astype(np.float32)
print(train_data)
train_x, train_y = train_data[:, :1], train_data[:, 1:]
test_data = np.loadtxt("test.txt").astype(np.float32)
test_x, test_y = test_data[:, :1], test_data[:, 1:]

# def loss
def my_loss_fn(_x, _y):
    return tf.reduce_mean((_x - _y) ** 2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

# def optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer=opt, loss=my_loss_fn)
model.fit(train_x, train_y, epochs=5000, batch_size=1024)
pred_y = model.predict(test_x)

# Draw the graphs
plt.plot(test_x, test_y, "o")
plt.plot(test_x, pred_y, "v")
plt.show()