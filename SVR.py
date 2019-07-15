import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import TimeSeriesPipeline as timeseries
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import tensorflow as tf

'''
Energy consume regression by SVR
'''

data = pd.read_csv('energydata.csv')
ibm = pd.read_csv('ibm_stock.csv')

'''
stock = ibm.iloc[:, 1].values.reshape(-1, 1)
time_step = 7
x_data, y_data = timeseries.TimeSeriesData(stock, stock, time_step)
print(x_data.shape)
scaler = MinMaxScaler()
#x_data = scaler.fit_transform(x_data)
#y_data = scaler.fit_transform(y_data)

output_dim = 50
x_data = x_data.reshape(len(x_data), time_step, 1)
model = Sequential()
model.add(
        keras.layers.LSTM(units = output_dim, # output neurons of this LSTM layer
                          input_shape = (time_step, 1) # input of an LSTM cell: [time step, features]
        ))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
history = model.fit(x_data, y_data, batch_size = 10, epochs = 100)
'''

y = data['Appliances'] # predict target: energy
x = data.iloc[:, 2:]
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
time_step = 7
y = y.values.reshape(-1, 1)
x, y = timeseries.TimeSeriesData(y, y, time_step)
train_end = int(len(x)*0.01)
x_train, y_train = x[0:train_end], y[0:train_end]
x_test, y_test = x[train_end:], y[train_end:]

#Data scaling
scaler = MinMaxScaler()
#scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)

output_dim = 250
#x_train = x_train.reshape(len(x_train), time_step, 1)

'''
model = Sequential()
model.add(
        keras.layers.LSTM(units = output_dim, # output neurons of this LSTM layer
                          input_shape = (time_step, 1) # input of an LSTM cell: [time step, features]
        ))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
history = model.fit(x_train, y_train, batch_size = 10, epochs = 3)
model.save('LSTM_h5')

model = tf.contrib.keras.models.load_model('LSTM_h5')
predict = model.predict(x_train)
y_train = scaler.inverse_transform(y_train)
predict = scaler.inverse_transform(predict)
plt.plot(y_train, color = 'red', label = 'original')
plt.plot(predict, color = 'blue', label = 'LSTM')
plt.legend()
plt.title('Train performance')
plt.savefig('Result.png')

plt.figure(2)
x_test = scaler.fit_transform(x_test)
x_test = x_test.reshape(len(x_test), time_step, 1)
predict = model.predict(x_test)
y_test = scaler.fit_transform(y_test)
loss, mse = model.evaluate(x_test, y_test)
#mse = scaler.inverse_transform(mse)
predict = scaler.inverse_transform(predict)

y_test = scaler.inverse_transform(y_test)
plt.plot(y_test, color = 'red', label = 'original')
plt.plot(predict, color = 'blue', label = 'LSTM')
plt.legend()
plt.title('Test performance  MSE: %s'%str(mse))
plt.savefig('Test result.png')

plt.show()

'''
svr = SVR(C = 10, gamma = 0.01)
svr.fit(x_train, y_train)
score = svr.score(x_train, y_train)
print('Score =', score)
predict = svr.predict(x_train).reshape(-1, 1)
predict = scaler.inverse_transform(predict)
y_train = scaler.inverse_transform(y_train)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(predict, color = 'blue', label = 'SVR predict')
#ax2 = fig.add_subplot(1, 21, 2)
ax.plot(y_train, color = 'red', label = 'Original')
ax.legend()
plt.show()

