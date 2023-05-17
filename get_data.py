import pandas as pd
from binance.client import Client
import ta

# Задайте свои учетные данные API Binance

# Создайте экземпляр клиента Binance API
client = Client()

# Задайте временной промежуток
start_time = '2016-01-01'
end_time = '2023-05-15'

# Получите исторические данные для пары торгов ETH/USDT в заданном временном промежутке
klines = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1DAY, start_time, end_time)

# Создайте списки для хранения данных
close_data = []
asks_data = []
bids_data = []
volume_data = []

# Итерируйтесь по полученным данным и извлекайте значения close, asks, bids и volume
for kline in klines:
    close = float(kline[4])
    asks = float(kline[2])
    bids = float(kline[3])
    volume = float(kline[5])
    close_data.append(close)
    asks_data.append(asks)
    bids_data.append(bids)
    volume_data.append(volume)

# Преобразуйте списки в объекты pd.Series
close_series = pd.Series(close_data)
asks_series = pd.Series(asks_data)
bids_series = pd.Series(bids_data)
volume_series = pd.Series(volume_data)

# Рассчитайте Bollinger Bands с использованием библиотеки ta
bollinger_bands = ta.volatility.BollingerBands(close_series)

# Извлеките значения верхней полосы Bollinger Bands
upper_band = bollinger_bands.bollinger_hband()

# Рассчитайте StochRSI с использованием библиотеки ta
stochrsi = ta.momentum.StochRSIIndicator(close_series).stochrsi()

# Создайте DataFrame с данными
data = pd.DataFrame({'close': close_data, 'asks': asks_data, 'bids': bids_data, 'volume':volume_data, 'volatile': upper_band, 'StochRSI': stochrsi})

# Сохраните DataFrame в файл ethusdt.csv
data.to_csv('ethusdt.csv', index=False)