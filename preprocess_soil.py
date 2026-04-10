import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading scaler objects

# Load dataset
csv_path = "Data/Soil/soil_data.csv"
data = pd.read_csv(csv_path)

# Separate features and target
X = data.drop(columns=["Output"])  # Features
y = data["Output"]  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data and scaler
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("Data/Soil/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("Data/Soil/X_test_scaled.csv", index=False)
pd.DataFrame(y_train).to_csv("Data/Soil/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("Data/Soil/y_test.csv", index=False)
joblib.dump(scaler, "Models/Soil_Analysis/scaler.pkl")
