import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from preprocess_irrigation import preprocess_data #type: ignore


def train_model(data_file):
    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data(data_file)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save trained model
    joblib.dump(model, 'models/irrigation_advice/irrigation_model.pkl')
    
    return model

if __name__ == "__main__":
    train_model("data/Irrigation/Irrigation_Dataset.csv")
