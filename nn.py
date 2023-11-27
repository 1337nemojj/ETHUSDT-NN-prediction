import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.models import load_model

data = pd.read_csv('ethusdt.csv')  
prices = data['close'].values.reshape(-1, 1)  
volume = data['volume'].values.reshape(-1, 1)  
StochRSI = data['StochRSI'].values.reshape(-1, 1)  
asks = data['asks'].values.reshape(-1,1)
bids = data['bids'].values.reshape(-1,1)
volatile = data['volatile'].values.reshape(-1,1)

scaler_volatile = MinMaxScaler(feature_range=(0, 1))
volatile_scaled = scaler_volatile.fit_transform(volatile)
scaler_asks= MinMaxScaler(feature_range=(0, 1))
asks_scaled = scaler_asks.fit_transform(asks)
scaler_bids = MinMaxScaler(feature_range=(0, 1))
bids_scaled = scaler_bids.fit_transform(bids)
scaler_price = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler_price.fit_transform(prices)
scaler_volume = MinMaxScaler(feature_range=(0, 1))
volume_scaled = scaler_volume.fit_transform(volume)
scaler_StochRSI = MinMaxScaler(feature_range=(0, 1))
StochRSI_scaled = scaler_StochRSI.fit_transform(StochRSI)


train_size = int(len(prices_scaled) * 0.8)
test_size = len(prices_scaled) - train_size
train_data = prices_scaled[:train_size, :]
test_data = prices_scaled[train_size:, :]
train_volume = volume_scaled[:train_size, :]
test_volume = volume_scaled[train_size:, :]
train_StochRSI = StochRSI_scaled[:train_size, :]
test_StochRSI = StochRSI_scaled[train_size:, :]
train_asks = asks_scaled[:train_size, :]
test_asks = asks_scaled[train_size:, :]
train_bids = bids_scaled[:train_size, :]
test_bids = bids_scaled[train_size:, :]
train_volatile = volatile_scaled[:train_size, :]
test_volatile = volatile_scaled[train_size:, :]

def create_sequences(data, volume, StochRSI, asks, bids,volatile, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(np.concatenate((data[i:i + seq_length, :], volume[i:i + seq_length, :], StochRSI[i:i + seq_length, :], asks[i:i + seq_length, :], bids[i:i + seq_length, :], volatile[i:i + seq_length, :]), axis=1))
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train_data, train_volume, train_StochRSI, train_asks, train_bids,train_volatile, seq_length)
X_test, y_test = create_sequences(test_data, test_volume, test_StochRSI,test_asks, test_bids,test_volatile, seq_length)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 6))) 
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16)

predicted_prices = model.predict(X_test)
predicted_prices = scaler_price.inverse_transform(predicted_prices)

for price in predicted_prices:
    print(price[0])

y_test = np.array(y_test).reshape(-1, 1)
actual_prices = scaler_price.inverse_transform(y_test)

plt.plot(actual_prices, label='Фактическая цена')
plt.plot(predicted_prices, label='Предсказанная цена')
plt.xlabel('Временные шаги')
plt.ylabel('Цена')
plt.title('Сравнение фактической и предсказанной цены')
plt.legend()
plt.show()