import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load data
train_data = np.loadtxt("train3d.txt").astype(np.float32)
train_x, train_y = train_data[:, :2], train_data[:, 2:]
test_data = np.loadtxt("test3d.txt").astype(np.float32)
test_x, test_y = test_data[:, :2], test_data[:, 2:]

# def loss
def my_loss_fn(_x, _y):
    return tf.reduce_mean((_x - _y) ** 2)

# Hyperparameters
layer_sizes = [1] + [128] * 4 + [1]

model = tf.keras.Sequential()
for units in layer_sizes[1:-1]:
    model.add(tf.keras.layers.Dense(units, activation="relu"))
model.add(tf.keras.layers.Dense(1))
# def optimizer with learning rate of 0.001
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss=my_loss_fn)
model.fit(train_x, train_y, epochs=2000, batch_size=2048)
model.summary()
pred_y = model.predict(test_x)

ax = plt.axes(projection='3d')
ax.scatter3D(test_x[:,:1], test_x[:,1:], pred_y, cmap='red')
ax.scatter3D(test_x[:,:1], test_x[:,1:], test_y, cmap='greens')
plt.show()
