
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model("model/lstm_model.h5")
scaler = MinMaxScaler()
scaler.min_ = np.load("model/scaler_min.npy")
scaler.scale_ = 1 / (np.load("model/scaler_max.npy") - scaler.min_)

def prediksi_ai(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    if len(data) < 10:
        return "0000"
    input_seq = np.array(data[-10:])
    input_scaled = scaler.transform(input_seq).reshape(1, 10, 4)
    pred = model.predict(input_scaled)[0]
    pred_digits = scaler.inverse_transform([pred])[0]
    return ''.join([str(int(round(d))) for d in pred_digits])
