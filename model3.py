import math

import finplot as fplt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

past_size = 30

stock_data = pd.read_csv("input_data.csv", parse_dates=["date"], engine="c")

fplt.create_plot()
fplt.background = '#ff0'

fplt.plot(stock_data['date'], stock_data["close"], legend='actual')

close_prices = stock_data['close']
values = close_prices.values
training_data_len = math.ceil(len(values) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values.reshape(-1, 1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(past_size, len(train_data)):
    x_train.append(train_data[i - past_size:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[training_data_len - past_size:, :]
x_test = []
y_test = values[training_data_len:]

for i in range(past_size, len(test_data)):
    x_test.append(test_data[i - past_size:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pv = np.zeros((148, 1))

for i in range(3):
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    print(model.summary())

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=10)

    predictions = model.predict(x_test)
    print(predictions.shape)
    pv += predictions

predictions = pv / 3
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(rmse)

data = stock_data
train = data[:training_data_len]
validation = data[training_data_len:]
validation['predictions'] = predictions
validation.reset_index(inplace=True, drop=True)
fplt.plot(validation['date'], validation["predictions"], legend='predicted')

validation.to_csv("predicted_day_data.csv", index=False)

fplt.show()
