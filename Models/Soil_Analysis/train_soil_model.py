import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed data
X_train = pd.read_csv("Data/Soil/X_train_scaled.csv").values
X_test = pd.read_csv("Data/Soil/X_test_scaled.csv").values
y_train = pd.read_csv("Data/Soil/y_train.csv").values.ravel()
y_test = pd.read_csv("Data/Soil/y_test.csv").values.ravel()

# Define the neural network
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)

# Save the trained model
model.save("Models/Soil_Analysis/soil_tabular_model.h5")
