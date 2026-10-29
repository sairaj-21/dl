import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed

def generate_series(n=1000):
    t = np.arange(n)
    return 10 + 0.01 * t + np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(n)

series = generate_series(1000)
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

INPUT_WINDOW, OUTPUT_WINDOW = 30, 10

def make_windows(data, input_w, output_w):
    X, Y = [], []
    for i in range(len(data) - input_w - output_w):
        X.append(data[i:i + input_w])
        Y.append(data[i + input_w:i + input_w + output_w])
    return np.array(X), np.array(Y)

X, Y = make_windows(series_scaled, INPUT_WINDOW, OUTPUT_WINDOW)
X, Y = X[..., np.newaxis], Y[..., np.newaxis]

split = int(0.8 * len(X))
X_train, X_test, Y_train, Y_test = X[:split], X[split:], Y[:split], Y[split:]

encoder_inputs = Input(shape=(INPUT_WINDOW, 1))
_, state_h, state_c = LSTM(64, return_state=True)(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(OUTPUT_WINDOW, 1))
decoder_outputs, _, _ = LSTM(64, return_sequences=True, return_state=True)(decoder_inputs, initial_state=encoder_states)
decoder_outputs = TimeDistributed(Dense(1))(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

decoder_input_train, decoder_input_test = np.zeros_like(Y_train), np.zeros_like(Y_test)

model.fit([X_train, decoder_input_train], Y_train, validation_data=([X_test, decoder_input_test], Y_test), epochs=20, batch_size=32, verbose=1)

preds = model.predict([X_test, decoder_input_test])
preds_unscaled = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
Y_test_unscaled = scaler.inverse_transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)

mae = mean_absolute_error(Y_test_unscaled.flatten(), preds_unscaled.flatten())
rmse = np.sqrt(mean_squared_error(Y_test_unscaled.flatten(), preds_unscaled.flatten()))
print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")