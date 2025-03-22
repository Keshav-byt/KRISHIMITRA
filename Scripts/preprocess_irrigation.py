import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(input_file):
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Encode categorical features
    Crop_type_enc = LabelEncoder()
    df['Soil Colour'] = Crop_type_enc.fit_transform(df['Soil Colour'])
    df['Label'] = Crop_type_enc.fit_transform(df['Crop type'])
    
    # Save encoders for later use
    joblib.dump(Crop_type_enc, 'models/Irrigation_Advice/Crop_type.pkl')
    
    # Select features and target
    X = df[['Soil Colour', 'Ph', 'Soil Moisture', 'Temperature', 'Air Humidity']]
    y = df['Crop type']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for inference
    joblib.dump(scaler, 'models/Irrigation_advice/scaler.pkl')
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data("data/Irrigation/Irrigation_Dataset.csv")
