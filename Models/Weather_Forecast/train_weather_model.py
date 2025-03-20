import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import joblib  # For saving the scaler
import os

# Step 1: Load the dataset
data = pd.read_csv("Data/Weather/weather_data.csv")

# Step 2: Preprocess the data
# Check for missing values
if data.isnull().sum().any():
    data = data.dropna()  # Drop rows with missing values, or use imputation methods

# Scaling the data for LSTM (Temperature, Humidity, WindSpeed)
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale only the relevant columns (Temperature, Humidity, WindSpeed)
scaled_data = scaler.fit_transform(data[['Temperature', 'Humidity', 'WindSpeed']])

# Step 3: Prepare the dataset for LSTM
# Create dataset with time steps (look-back)
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])  # Collecting previous n timesteps
        y.append(data[i + time_step, 0])  # Target is the temperature (index 0)
    return np.array(X), np.array(y)

time_step = 10  # Number of previous time steps to look at for prediction
X, y = create_dataset(scaled_data, time_step)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 1: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # Output layer (predicting temperature)

# Step 2: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 3: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Save the model
model.save('weather_model.h5')
print("Model saved successfully to weather_model.h5")

# Step 7: Evaluate the model and make predictions
predictions = model.predict(X_test)

# Inverse transform predictions to the original scale
# Concatenate predictions with zeros for the non-predicted columns (Humidity, WindSpeed)
# We add zeros only for the columns that were not predicted, i.e., Humidity and WindSpeed.
predictions_rescaled = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, 0]

# Inverse transform actual test data to the original scale
y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2))), axis=1))[:, 0]

# Step 9: Evaluate model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test_actual, predictions_rescaled)
mae = mean_absolute_error(y_test_actual, predictions_rescaled)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Step 10: Save the model and scaler
# Ensure the directory for saving the model exists
model_dir = 'Models/Weather_Forecast'
os.makedirs(model_dir, exist_ok=True)

# Save the LSTM model to .h5 file
model.save(os.path.join(model_dir, 'weather_model.h5'))
print("Model saved successfully to weather_model.h5")

# Save the scaler to .pkl file
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
print("Scaler saved successfully to scaler.pkl")
