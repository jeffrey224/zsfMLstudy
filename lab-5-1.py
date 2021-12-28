import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# from google.colab import files

from keras.preprocessing import image


import zipfile

# Directory with our training horse pictures
train_horse_dir = os.path.join('./tmp/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('./tmp/horse-or-human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('./tmp/validation-horse-or-human/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('./tmp/validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# # Parameters for our graph; we'll output images in a 4x4 configuration
# nrows = 4
# ncols = 4
#
# # Index for iterating over images
# pic_index = 0
#
# # Set up matplotlib fig, and size it to fit 4x4 pics
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)
#
# pic_index += 8
# next_horse_pix = [os.path.join(train_horse_dir, fname)
#                 for fname in train_horse_names[pic_index-8:pic_index]]
# next_human_pix = [os.path.join(train_human_dir, fname)
#                 for fname in train_human_names[pic_index-8:pic_index]]
#
# for i, img_path in enumerate(next_horse_pix+next_human_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)
#
#   img = mpimg.imread(img_path)
#   plt.imshow(img)
#
# plt.show()


print(tf.__version__)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
  #  tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
 #   tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()



model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=72,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './tmp/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

history = model.fit(
      train_generator,
      validation_data = validation_generator,
      epochs=15,
      steps_per_epoch=8,
      validation_steps=8,
      verbose=2)


#save model

model.save('./savedmodels/my_model')


#
# uploaded = files.upload()
#
# for fn in uploaded.keys():

    # predicting images
    # path = './content/horse5-235.png'


