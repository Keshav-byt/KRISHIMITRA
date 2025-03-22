from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError #type: ignore
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import os
from PIL import Image
import io
import traceback
import importlib.util
import sys
import requests

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
LABEL_ENC = None
API_KEY = "38951a8de7c22843f1f124e445f7b55c" # OpenWeatherMap API key

current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = current_dir
project_root = os.path.dirname(current_dir) 

# Standardize paths
PATHS = {
    'soil_model': os.path.join(project_root, "Models", "Soil_Analysis", "soil_tabular_model.h5"),
    'soil_scaler': os.path.join(project_root, "Models", "Soil_Analysis", "scaler.pkl"),
    'weather_model': os.path.join(project_root, "Models", "Weather_Forecast", "weather_model.h5"),
    'weather_scaler': os.path.join(project_root, "Models", "Weather_Forecast", "scaler.pkl"),
    'pest_model': os.path.join(project_root, "Models", "Pest_Detection", "pest_detection_model.h5"),
    'irrigation_model': os.path.join(project_root, "Models", "Irrigation_Advice", "crop_recommend.pkl"),
    'irrigation_scaler': os.path.join(project_root, "Models", "Irrigation_Advice", "scaler.pkl"),
    'crop_type_enc': os.path.join(project_root, "Models", "Irrigation_Advice", "crop_encoder.pkl"),
    'soil_colour_enc': os.path.join(project_root, "Models", "Irrigation_Advice", "soil_color_encoder.pkl"),
    'preprocess_soil': os.path.join(scripts_dir, "preprocess_soil.py"),
    'preprocess_irrigation': os.path.join(scripts_dir, "preprocess_irrigation.py"),
    'train_soil': os.path.join(project_root, "Models", "Soil_Analysis", "train_soil_model.py"),
    'train_weather': os.path.join(project_root, "Models", "Weather_Forecast", "train_weather_model.py"),
    'train_pest': os.path.join(project_root, "Models", "Pest_Detection", "train_pest_model.py"),
    'train_irrigation': os.path.join(project_root, "Models", "Irrigation_Advice", "train_irrigation_model.py"),
}

def ensure_directory_exists(file_path):
    """Ensure the directory for a file exists."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def get_weather(city):
    """Get weather data for a city using OpenWeatherMap API."""
    try:
        URL = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(URL)
        if response.status_code == 200:
            data = response.json()
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            return temperature, humidity
        else:
            logger.error(f"Weather API error: {response.status_code} - {response.text}")
            return None, None
    except Exception as e:
        logger.error(f"Weather API request failed: {e}")
        return None, None

# Load models and scalers
def load_models_and_scalers():
    global PEST_MODEL, LABEL_ENC
    models = {}
    scalers = {}

    # Ensure directories exist
    for path in PATHS.values():
        ensure_directory_exists(path)

    # Run models again, if necessary
    # Soil preprocessing
    if (not os.path.exists(PATHS['soil_scaler']) or 
        not os.path.exists(os.path.join(project_root, "data", "soil", "X_train_scaled.csv"))):
        logger.info("Processing Soil Data...")
        preprocess = import_module_from_file(PATHS['preprocess_soil'], "preprocess")
        if preprocess:
            logger.info("Soil Data Processed.")

    # Irrigation preprocessing
    if (not os.path.exists(PATHS['irrigation_scaler']) or
        not os.path.exists(PATHS['crop_type_enc'])):
        logger.info("Processing Irrigation Data...")
        preprocess_irrigation = import_module_from_file(PATHS['preprocess_irrigation'], "preprocess_irrigation")
        if preprocess_irrigation:
            logger.info("Irrigation Data Processed.")
    
    # Training models if necessary
    if not os.path.exists(PATHS['soil_model']):
        logger.info("Training soil model...")
        train_soil = import_module_from_file(PATHS['train_soil'], "train_soil")
        if train_soil:
            logger.info("Soil model training completed")
    
    if not os.path.exists(PATHS['weather_model']) or not os.path.exists(PATHS['weather_scaler']):
        logger.info("Training weather model...")
        train_weather = import_module_from_file(PATHS['train_weather'], "train_weather")
        if train_weather:
            logger.info("Weather model training completed")
        
    if not os.path.exists(PATHS['pest_model']):
        logger.info("Training pest detection model...")
        train_pest = import_module_from_file(PATHS['train_pest'], "train_pest")
        if train_pest:
            logger.info("Pest detection model training completed")

    if not os.path.exists(PATHS['irrigation_model']) or not os.path.exists(PATHS['irrigation_scaler']):
        logger.info("Training irrigation advice model...")
        train_irrigation = import_module_from_file(PATHS['train_irrigation'], "train_irrigation")
        if train_irrigation:
            logger.info("Irrigation advice model training completed")

    try:
        # Load Soil Analysis Model
        if os.path.exists(PATHS['soil_model']):
            models['soil'] = load_model(PATHS['soil_model'])
            logger.info("Soil analysis model loaded successfully")
        else:
            logger.error(f"Soil model not found at {PATHS['soil_model']}")

        # Load Soil Analysis Scaler
        if os.path.exists(PATHS['soil_scaler']):
            scalers['soil'] = joblib.load(PATHS['soil_scaler'])
            logger.info("Soil scaler loaded successfully")
        else:
            logger.error(f"Soil scaler not found at {PATHS['soil_scaler']}")

        # Load Weather Forecast Model
        if os.path.exists(PATHS['weather_model']):
            models['weather'] = load_model(
                PATHS['weather_model'], 
                custom_objects={
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError()
                }
            )
            logger.info("Weather model loaded successfully")
        else:
            logger.error(f"Weather model not found at {PATHS['weather_model']}")

        # Load Weather Scaler
        if os.path.exists(PATHS['weather_scaler']):
            scalers['weather'] = joblib.load(PATHS['weather_scaler'])
            logger.info("Weather scaler loaded successfully")
        else:
            logger.error(f"Weather scaler not found at {PATHS['weather_scaler']}")
            
        # Load Pest Detection Model
        if os.path.exists(PATHS['pest_model']):
            PEST_MODEL = load_model(PATHS['pest_model'])
            logger.info("Pest detection model loaded successfully")
        else:
            logger.error(f"Pest detection model not found at {PATHS['pest_model']}")

        # Load Irrigation Advice Model
        if os.path.exists(PATHS['irrigation_model']):
            models['irrigation'] = joblib.load(PATHS['irrigation_model'])
            logger.info("Irrigation advice model loaded successfully")
        else:
            logger.error(f"Irrigation advice model not found at {PATHS['irrigation_model']}")

        # Load Irrigation Advice Scaler
        if os.path.exists(PATHS['irrigation_scaler']):
            scalers['irrigation'] = joblib.load(PATHS['irrigation_scaler'])
            logger.info("Irrigation advice scaler loaded successfully")
        else:
            logger.error(f"Irrigation advice scaler not found at {PATHS['irrigation_scaler']}")

        # Load Crop Type Encoder
        if os.path.exists(PATHS['crop_type_enc']):
            LABEL_ENC = joblib.load(PATHS['crop_type_enc'])
            logger.info("Crop type encoder loaded successfully")
        else:
            logger.error(f"Crop type encoder not found at {PATHS['crop_type_enc']}")

        # Load Soil Color Encoder
        if os.path.exists(PATHS['soil_colour_enc']):
            models['soil_colour_enc'] = joblib.load(PATHS['soil_colour_enc'])
            logger.info("Soil color encoder loaded successfully")
        else:
            logger.error(f"Soil color encoder not found at {PATHS['soil_colour_enc']}")

    except Exception as e:
        logger.error(f"Error loading models/scalers: {e}")
        logger.error(traceback.format_exc())

    return models, scalers

# Load models and scalers at startup
MODELS, SCALERS = load_models_and_scalers()

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
        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({"error": f"Missing required soil features: {', '.join(missing)}"}), 400

        # Validate numeric values
        for feature in feature_names:
            try:
                value = float(data[feature])
                data[feature] = value  # Ensure numeric conversion
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {feature}: must be numeric"}), 400

        # Prepare features
        features = [data[feature] for feature in feature_names]
        features_array = np.array([features])

        # Scale features
        scaled_features = SCALERS['soil'].transform(features_array)

        # Make prediction
        prediction = MODELS['soil'].predict(scaled_features)
        
        # Convert prediction to interpretable result
        prediction_value = float(prediction[0][0])
        fertility_status = "High" if prediction_value > 0.5 else "Low"
        
        # Calculate confidence
        confidence = abs(prediction_value - 0.5) * 200  # Scale to 0-100%
        
        # Get recommendations based on actual soil values
        recommendations = get_soil_recommendation(data, fertility_status)
        
        # Return detailed response
        return jsonify({
            "fertility_status": fertility_status,
            "confidence": round(confidence, 2),
            "recommendation": recommendations,
        }), 200

    except Exception as e:
        logger.error(f"Soil analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error during soil analysis: {str(e)}"}), 500

def get_soil_recommendation(soil_data, fertility_status):
    """Generate soil fertility recommendations based on soil data"""
    recommendations = []
    
    # Detailed recommendations based on NPK values
    if soil_data["N"] < 140:
        recommendations.append("Nitrogen levels are very low. Apply nitrogen-rich fertilizers like urea or ammonium sulfate.")
    elif soil_data["N"] < 200:
        recommendations.append("Nitrogen levels are moderately low. Consider adding nitrogen-rich organic matter or balanced NPK fertilizer.")
    
    if soil_data["P"] < 25:
        recommendations.append("Phosphorus levels are very low. Apply phosphate fertilizers like superphosphate or rock phosphate.")
    elif soil_data["P"] < 40:
        recommendations.append("Phosphorus levels are moderately low. Consider adding bone meal or balanced fertilizer with higher P content.")
    
    if soil_data["K"] < 180:
        recommendations.append("Potassium levels are very low. Apply potash fertilizers like muriate of potash or sulfate of potash.")
    elif soil_data["K"] < 250:
        recommendations.append("Potassium levels are moderately low. Consider adding wood ash or balanced fertilizer with higher K content.")
    
    # pH recommendations
    if soil_data["pH"] < 5.5:
        recommendations.append("Soil is strongly acidic. Add agricultural lime to raise pH and improve nutrient availability.")
    elif soil_data["pH"] < 6.0:
        recommendations.append("Soil is moderately acidic. Consider adding dolomitic lime to gradually raise pH.")
    elif soil_data["pH"] > 8.0:
        recommendations.append("Soil is strongly alkaline. Add organic matter or gypsum to gradually lower pH.")
    elif soil_data["pH"] > 7.5:
        recommendations.append("Soil is moderately alkaline. Incorporate more organic matter to improve soil structure and lower pH.")
    
    # Organic carbon recommendations
    if soil_data["OC"] < 0.4:
        recommendations.append("Organic carbon is very low. Add compost, manure or practice crop rotation with cover crops.")
    elif soil_data["OC"] < 0.75:
        recommendations.append("Organic carbon is low. Incorporate more organic matter and reduce tillage if possible.")
    
    if fertility_status == "Low" and not recommendations:
        recommendations.append("Consider adding organic compost to improve overall soil fertility.")
        
    return recommendations

@app.route("/weather-prediction", methods=["POST"])
def weather_prediction():
    """Weather prediction endpoint"""
    try:
        # Validate model is loaded
        if 'weather' not in MODELS or 'weather' not in SCALERS:
            return jsonify({"error": "Weather prediction model not loaded"}), 500

        # Extract features from request
        data = request.json
        if not data:
            return jsonify({"error": "No weather data provided"}), 400
            
        if isinstance(data, dict):  # If data is a dictionary, extract "data"
            data = data.get("data", None)
        
        if not isinstance(data, list):
            return jsonify({"error": "Invalid input format. Expected a JSON array or object with 'data' property."}), 400

        # Validate the input size
        if len(data) != 3:
            return jsonify({"error": f"Invalid input size. Expected exactly 3 values, got {len(data)}."}), 400

        # Validate numeric values
        try:
            data = [float(val) for val in data]
        except (ValueError, TypeError):
            return jsonify({"error": "All input values must be numeric"}), 400

        # Convert to numpy array and reshape for the model input
        features = np.array([data])
        scaled_features = SCALERS['weather'].transform(features)
        
        # Reshape for LSTM input
        scaled_features = scaled_features.reshape(1, 1, 3)
        
        # Make prediction
        scaled_prediction = MODELS['weather'].predict(scaled_features)
        
        # Inverse transform prediction to the original scale
        temp_pred = scaled_prediction[0][0]
        dummy_features = np.zeros((1, 3))
        dummy_features[0, 0] = temp_pred  # Set the first column (temperature) to our prediction
        
        # Inverse transform to get the actual temperature
        original_scale_prediction = SCALERS['weather'].inverse_transform(dummy_features)[0, 0]

        # Return the predicted temperature
        return jsonify({
            "predicted_temperature": round(float(original_scale_prediction), 2)
        }), 200

    except Exception as e:
        logger.error(f"Weather prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error during weather prediction: {str(e)}"}), 500
    
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
        
        # Validate file type
        if not image_file.filename or '.' not in image_file.filename:
            return jsonify({"error": "Invalid image file"}), 400
            
        extension = image_file.filename.rsplit('.', 1)[1].lower()
        if extension not in {'png', 'jpg', 'jpeg', 'gif'}:
            return jsonify({"error": "File extension not allowed"}), 400

        try:
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
            image = image.resize((224, 224))  # Resize to match model input size
            image_array = np.array(image) / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

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
            "confidence": round(confidence, 2)
        }
        
        # Add class name if available
        if class_names and predicted_class < len(class_names):
            result["class_name"] = class_names[predicted_class]
        else:
            result["class_id"] = int(predicted_class)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Pest detection error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error during pest detection: {str(e)}"}), 500
    
@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    """Crop recommendation endpoint"""
    try:
        # Check if required models and encoders are loaded
        if 'irrigation' not in MODELS or 'irrigation' not in SCALERS or LABEL_ENC is None:
            return jsonify({"error": "Crop prediction model not loaded"}), 500
            
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Validate required fields
        required_fields = ["soil_color", "city"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
        
        soil_color = data.get("soil_color")
        city = data.get("city")
        
        # Validate soil_color is in known classes
        if 'soil_colour_enc' not in MODELS:
            return jsonify({"error": "Soil color encoder not loaded"}), 500
            
        soil_color_encoder = MODELS['soil_colour_enc']
        
        # Check if the soil color is valid
        try:
            if not hasattr(soil_color_encoder, 'classes_') or soil_color not in soil_color_encoder.classes_:
                return jsonify({"error": f"Invalid soil_color. Supported values: {list(soil_color_encoder.classes_) if hasattr(soil_color_encoder, 'classes_') else 'unknown'}"}), 400
        except Exception as e:
            logger.error(f"Error validating soil color: {e}")
            return jsonify({"error": f"Error validating soil color: {str(e)}"}), 500
                
        # Get weather data
        temperature, humidity = get_weather(city)
        if temperature is None or humidity is None:
            return jsonify({"error": f"Could not fetch weather data for city: {city}"}), 400
        
        try:
            # Encode categorical features
            soil_color_encoded = soil_color_encoder.transform([soil_color])[0]
            
            # The model expects 3 features: soil color, temperature, humidity
            input_data = np.array([[soil_color_encoded, temperature, humidity]])
            input_scaled = SCALERS['irrigation'].transform(input_data)
        except Exception as e:
            logger.error(f"Error processing input data: {e}")
            return jsonify({"error": f"Error processing input data: {str(e)}"}), 400
        
        # Predict crop type
        prediction = MODELS['irrigation'].predict(input_scaled)
        crop_type = LABEL_ENC.inverse_transform(prediction)[0]
        
        # Get crop suitability score (using prediction probabilities)
        probabilities = MODELS['irrigation'].predict_proba(input_scaled)[0]
        max_probability = float(np.max(probabilities) * 100)
        
        # Add recommendations based on predicted crop
        recommendations = get_crop_recommendations(crop_type, temperature, humidity)
        
        return jsonify({
            "city": city,
            "temperature": temperature,
            "humidity": humidity,
            "recommended_crop": crop_type
        }), 200
    
    except Exception as e:
        logger.error(f"Crop prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error during crop prediction: {str(e)}"}), 500

def get_crop_recommendations(crop_type, temperature, humidity):
    """Generate crop-specific recommendations based on weather conditions"""
    recommendations = []
    
    # General temperature recommendations
    if temperature < 15:
        recommendations.append(f"Current temperature ({temperature}°C) is low for {crop_type}. Consider providing additional protection.")
    elif temperature > 35:
        recommendations.append(f"Current temperature ({temperature}°C) is high for {crop_type}. Ensure adequate irrigation.")
    
    # General humidity recommendations
    if humidity < 30:
        recommendations.append(f"Current humidity ({humidity}%) is low. Consider increasing irrigation frequency.")
    elif humidity > 80:
        recommendations.append(f"Current humidity ({humidity}%) is high. Watch for fungal diseases.")
    
    # Crop-specific recommendations
    if crop_type.lower() == "barley":
        recommendations.append("Barley grows best in well-drained soil with moderate moisture.")
        if temperature > 30:
            recommendations.append("Barley is sensitive to high temperatures. Consider shade or increased irrigation.")
    elif crop_type.lower() == "wheat":
        recommendations.append("Wheat requires full sun and moderate water supply.")
        if humidity > 70:
            recommendations.append("High humidity may increase risk of wheat diseases. Consider fungicide application.")
    elif crop_type.lower() == "rice":
        recommendations.append("Rice requires flooding or consistent moisture.")
        if temperature < 20:
            recommendations.append("Low temperatures may delay rice growth. Consider delayed planting.")
    
    # If no specific recommendations are given
    if not recommendations:
        recommendations.append(f"Monitor water needs for {crop_type} based on local conditions.")
    
    return recommendations

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)