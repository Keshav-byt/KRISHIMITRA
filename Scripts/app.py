from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError # type: ignore
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import os
from PIL import Image
import io
import traceback
import importlib.util
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Create Flask app
app = Flask(__name__)
CORS(app)

# Global variables
MODELS = {}
SCALERS = {}
PEST_MODEL = None
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = current_dir
project_root = os.path.dirname(current_dir)  # Go up one level from Scripts

# Load models and scalers
def load_models_and_scalers():
    global PEST_MODEL  # Add global declaration for PEST_MODEL
    models = {}
    scalers = {}

    # Model and Scaler Paths
    soil_model_path = os.path.join(project_root, "Models", "Soil_Analysis", "soil_tabular_model.h5")
    soil_scaler_path = os.path.join(project_root, "Models", "Soil_analysis", "scaler.pkl")
    weather_model_path = os.path.join(project_root, "Models", "Weather_Forecast", "weather_model.h5")
    weather_scaler_path = os.path.join(project_root, "Models", "Weather_Forecast", "scaler.pkl")
    pest_model_path = os.path.join(project_root, "Models", "Pest_Detection", "pest_detection_model.h5")

    preprocess_path = os.path.join(scripts_dir, "preprocess_soil.py")
    train_soil_path = os.path.join(project_root, "Models", "Soil_analysis", "train_soil_model.py")
    train_weather_path = os.path.join(project_root, "Models", "Weather_Forecast", "train_weather_model.py")
    train_pest_path = os.path.join(project_root, "Models", "Pest_Detection", "train_pest_model.py")
    
    # Run preprocessing if necessary
    if (not os.path.exists(soil_scaler_path) or 
        not os.path.exists(os.path.join(project_root, "data", "soil", "X_train_scaled.csv"))):
        logger.info("Processing Soil Data...")
        try:
            if os.path.exists(preprocess_path):
                preprocess = import_module_from_file(preprocess_path, "preprocess")
                logger.info("Soil Data Processed.")
            else:
                logger.error(f"Preprocessing script not found at {preprocess_path}")
        except Exception as e:
            logger.error(f"Error in soil preprocessing: {e}")
            logger.error(traceback.format_exc())
    
    # Run soil model training if necessary
    if not os.path.exists(soil_model_path):
        logger.info("Training soil model...")
        try:
            if os.path.exists(train_soil_path):
                train_soil = import_module_from_file(train_soil_path, "train_soil")
                logger.info("Soil model training completed")
            else:
                logger.error(f"Soil training script not found at {train_soil_path}")
        except Exception as e:
            logger.error(f"Error in soil model training: {e}")
            logger.error(traceback.format_exc())
    
    # Run weather model training if necessary
    if not os.path.exists(weather_model_path) or not os.path.exists(weather_scaler_path):
        logger.info("Training weather model...")
        try:
            if os.path.exists(train_weather_path):
                train_weather = import_module_from_file(train_weather_path, "train_weather")
                logger.info("Weather model training completed")
            else:
                logger.error(f"Weather training script not found at {train_weather_path}")
        except Exception as e:
            logger.error(f"Error in weather model training: {e}")
            logger.error(traceback.format_exc())
        
    # Load pest detection model
    if not os.path.exists(pest_model_path):
        logger.info("Training pest detection model...")
        try:
            if os.path.exists(train_pest_path):
                train_pest = import_module_from_file(train_pest_path, "train_pest")
                logger.info("Pest detection model training completed")
            else:
                logger.error(f"Pest detection training script not found at {train_pest_path}")
        except Exception as e:
            logger.error(f"Error in pest detection model training: {e}")
            logger.error(traceback.format_exc())

    try:
        # Soil Analysis Model
        if os.path.exists(soil_model_path):
            models['soil'] = load_model(soil_model_path)
            logger.info(f"Soil analysis model loaded successfully from {soil_model_path}")
        else:
            logger.error(f"Soil model not found at {soil_model_path}")

        # Soil Analysis Scaler
        if os.path.exists(soil_scaler_path):
            scalers['soil'] = joblib.load(soil_scaler_path)
            logger.info(f"Soil scaler loaded successfully from {soil_scaler_path}")
        else:
            logger.error(f"Soil scaler not found at {soil_scaler_path}")

        # Weather Forecast Model
        if os.path.exists(weather_model_path):
            models['weather'] = load_model(
                weather_model_path, 
                custom_objects={
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError()
                }
            )
            logger.info(f"Weather model loaded successfully from {weather_model_path}")
        else:
            logger.error(f"Weather model not found at {weather_model_path}")

        # Weather Scaler
        if os.path.exists(weather_scaler_path):
            scalers['weather'] = joblib.load(weather_scaler_path)
            logger.info(f"Weather scaler loaded successfully from {weather_scaler_path}")
        else:
            logger.error(f"Weather scaler not found at {weather_scaler_path}")
            
        # Load Pest Detection Model
        if os.path.exists(pest_model_path):
            PEST_MODEL = load_model(pest_model_path)
            logger.info(f"Pest detection model loaded successfully from {pest_model_path}")
        else:
            logger.error(f"Pest detection model not found at {pest_model_path}")

    except Exception as e:
        logger.error(f"Error loading models/scalers: {e}")
        logger.error(traceback.format_exc())
        models = {}
        scalers = {}

    return models, scalers

# Load models and scalers at startup
MODELS, SCALERS = load_models_and_scalers()

@app.route("/", methods=["GET"])
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "soil_analysis": "soil" in MODELS,
            "weather_prediction": "weather" in MODELS,
            "pest_detection": PEST_MODEL is not None
        }
    }), 200

@app.route("/soil-analysis", methods=["POST"])
def soil_analysis():
    """Soil fertility analysis endpoint"""
    try:
        # Validate model is loaded
        if 'soil' not in MODELS or 'soil' not in SCALERS:
            return jsonify({"error": "Soil analysis model not loaded"}), 500

        # Extract features from request
        data = request.json
        if not data:
            return jsonify({"error": "No soil data provided"}), 400

        # Required feature names
        feature_names = ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"]
        
        # Validate input features
        if not all(feature in data for feature in feature_names):
            return jsonify({"error": "Missing required soil features"}), 400

        # Prepare features
        features = [data[feature] for feature in feature_names]
        features_array = np.array([features])

        # Scale features
        scaled_features = SCALERS['soil'].transform(features_array)

        # Make prediction
        prediction = MODELS['soil'].predict(scaled_features)
        
        # Convert prediction to interpretable result
        fertility_status = "High" if prediction[0][0] > 0.5 else "Low"
        confidence = float(prediction[0][0] * 100)

        return jsonify({
            "fertility_status": fertility_status,
            "confidence": confidence
        }), 200

    except Exception as e:
        logger.error(f"Soil analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error during soil analysis"}), 500

@app.route("/weather-prediction", methods=["POST"])
def weather_prediction():
    """Weather prediction endpoint"""
    try:
        # Validate model is loaded
        if 'weather' not in MODELS or 'weather' not in SCALERS:
            return jsonify({
                "error": "Weather prediction model or scaler not loaded. Please ensure the weather model and scaler files are available."
            }), 500

        # Extract features from request
        data = request.json
        if isinstance(data, dict):  # If data is a dictionary, extract "data"
            data = data.get("data", None)
        elif isinstance(data, list):  # If it's a list, treat it as raw input
            pass
        else:
            return jsonify({"error": "Invalid input format. Expected a JSON array or object."}), 400

        # Validate the input size (check for 3 values)
        if not data or len(data) != 3:
            return jsonify({
                "error": f"Invalid input size. Expected a JSON array with exactly 3 values, but got {len(data)} values."
            }), 400

        # Convert to numpy array and reshape to (1, 1, 3)
        input_array = np.array(data).reshape(1, 1, 3)

        # Normalize input using the scaler (Ensure correct input format)
        scaled_features = SCALERS['weather'].transform(input_array.reshape(-1, 3))
        scaled_features = scaled_features.reshape(1, 1, 3)

        # Make prediction
        prediction = MODELS['weather'].predict(scaled_features)

        # Return the predicted temperature
        return jsonify({
            "predicted_temperature": abs(float(prediction[0][0]))*100
        }), 200

    except Exception as e:
        logger.error(f"Weather prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error during weather prediction"}), 500
    
@app.route("/pest-detection", methods=["POST"])
def pest_detection():
    """Pest detection endpoint"""
    try:
        # Check if the pest model is loaded
        if PEST_MODEL is None:
            return jsonify({"error": "Pest detection model not loaded"}), 500

        # Check if an image was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = image.resize((224, 224))  # Resize to match model input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using model
        prediction = PEST_MODEL.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction) * 100)

        # Get class names (if available)
        class_names = []
        try:
            data_dir = os.path.join(project_root, "data", "Pest", "Processed_Images")
            if os.path.exists(data_dir):
                class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        except Exception as e:
            logger.warning(f"Could not get class names: {e}")

        result = {
            "predicted_class": int(predicted_class),
            "confidence": confidence
        }
        
        # Add class name if available
        if class_names and predicted_class < len(class_names):
            result["class_name"] = class_names[predicted_class]

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Pest detection error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error during pest detection: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)