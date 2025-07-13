
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(df):
    df = df.copy()
    df['Log_Volume'] = np.log(df['Volume'] + 1e-6)
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    price_features = ['Open', 'High', 'Low', 'Close', 'RSI', 'Log_Volume']
    df[price_features] = minmax_scaler.fit_transform(df[price_features])

    zscore_features = ['MACD', 'Returns']
    df[zscore_features] = standard_scaler.fit_transform(df[zscore_features])

    return df, minmax_scaler, standard_scaler
