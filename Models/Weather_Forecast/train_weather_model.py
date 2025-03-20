import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import joblib
import os

# Load the dataset
data = pd.read_csv("Data/Weather/weather_data.csv")

# Handle missing values
if data.isnull().sum().any():
    data = data.dropna()

# Scale relevant columns
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Temperature', 'Humidity', 'WindSpeed']])

# Create dataset with time steps (look-back)
def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])  # Sequence of time_step points
        y.append(data[i + time_step, 0])      # Target is next temperature
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build a more robust LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))  # Output layer (predicting temperature)

# Compile with appropriate loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Add early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the model
predictions = model.predict(X_test)

# Transform predictions and actual values back to original scale
def inverse_transform_temperature(scaled_values, scaler):
    # Create a dummy array with zeros for humidity and wind speed
    dummy = np.zeros((len(scaled_values), 3))
    dummy[:, 0] = scaled_values.flatten()  # Set first column (temperature)
    # Inverse transform
    return scaler.inverse_transform(dummy)[:, 0]

# Get temperature predictions and actual values in original scale
predictions_actual = inverse_transform_temperature(predictions, scaler)
y_test_actual = inverse_transform_temperature(y_test.reshape(-1, 1), scaler)

# Evaluate performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test_actual, predictions_actual)
mae = mean_absolute_error(y_test_actual, predictions_actual)
r2 = r2_score(y_test_actual, predictions_actual)

print(f'Mean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R² Score: {r2:.4f}')

# Save the model and scaler
model_dir = 'Models/Weather_Forecast'
os.makedirs(model_dir, exist_ok=True)

model.save(os.path.join(model_dir, 'weather_model.h5'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

print(f"Model and scaler saved to {model_dir}")