import tensorflow as tf
import numpy as np
import pandas as pd
from xlrd import open_workbook
book=open_workbook(r'./tmp/lab1-1data.xlsx')
sheet=book.sheets()[0]
ys=np.array([x.value for x in sheet.col(1,start_rowx=1)],dtype=float)

xs=np.array([x.value for x in sheet.col(0,start_rowx=1)],dtype=float)
# print(arr1)



sheet_a=pd.read_excel('./tmp/lab1-1data.xlsx')

sheet_b=sheet_a[:][1:2]

xx=np.array(sheet_a,dtype=float)
print(xx)
# xx=np.array([x.value for x in sheet.col(0,1,start_rowx=1)],dtype=float)
# print(xx)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256,input_shape=[1]),
    tf.keras.layers.Dense(512),

    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(16, activation='tanh'),
    # tf.keras.layers.Dense(32, activation='tanh'),
    # tf.keras.layers.Dense(64, activation='tanh'),
    # tf.keras.layers.Dense(128, activation='tanh'),
    # tf.keras.layers.Dense(512, activation='tanh'),
    # tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1)
])



# model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['accuracy'])
model.compile(optimizer='adam', loss='mean_squared_error')



model.fit(xs, ys, epochs=10)
print(model.predict([5.0,3.3,2.3,10.5,30.0]))