import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_input(data, window):
    X = []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
    return np.array(X)

def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)