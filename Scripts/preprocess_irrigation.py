import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(input_file):
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Encode categorical features
    soil_color_enc = LabelEncoder()
    df['Soil Colour Encoded'] = soil_color_enc.fit_transform(df['Soil Colour'])
    
    # Save encoders for later use
    joblib.dump(soil_color_enc, 'models/Irrigation_Advice/soil_color_encoder.pkl')
    
    # Select features and target
    X = df[['Soil Colour Encoded', 'Temperature', 'Air Humidity']]
    y = df['Crop type']
    
    # Encode target (crop types)
    crop_encoder = LabelEncoder()
    y_encoded = crop_encoder.fit_transform(y)
    
    # Save the crop encoder for inference
    joblib.dump(crop_encoder, 'models/Irrigation_Advice/crop_encoder.pkl')
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for inference
    joblib.dump(scaler, 'models/Irrigation_advice/scaler.pkl')
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data("data/Irrigation/Irrigation_Dataset.csv")
