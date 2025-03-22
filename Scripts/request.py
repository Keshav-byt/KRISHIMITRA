import requests
import json
import os
from pathlib import Path
import sys

# Flask base URL
BASE_URL = "http://127.0.0.1:5000"

def test_soil_fertility_classification():
    """Test the soil fertility analysis endpoint"""
    # Define different soil samples to test
    soil_samples = [
        {
            "name": "Good soil sample",
            "data": {
                "N": 280, "P": 80, "K": 350, "pH": 6.8,
                "EC": 1.5, "OC": 1.2, "S": 20, "Zn": 1.5,
                "Fe": 6.0, "Cu": 2.0, "Mn": 1.2, "B": 0.6
            }
        },
        {
            "name": "Poor soil sample",
            "data": {
                "N": 120, "P": 20, "K": 150, "pH": 5.2,
                "EC": 0.8, "OC": 0.3, "S": 8, "Zn": 0.5,
                "Fe": 2.0, "Cu": 0.5, "Mn": 0.4, "B": 0.2
            }
        }
    ]
    
    for sample in soil_samples:
        try:
            print(f"\nTesting soil sample: {sample['name']}")
            soil_response = requests.post(f"{BASE_URL}/soil-analysis", json=sample['data'])
            soil_response.raise_for_status()
            print("Soil Fertility Classification Response:")
            print(json.dumps(soil_response.json(), indent=2))
        except requests.exceptions.RequestException as e:
            print(f"Error during soil classification request: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON response for soil classification.")

def test_weather_prediction():
    """Test the weather prediction endpoint
    with different weather conditions"""
    # Test cases: [Temperature, Humidity, WindSpeed]
    weather_test_cases = [
        {"data": [32.0, 0.45, 15.0]},
        {"data": [25.0, 0.6, 10.0]},
        {"data": [5.0, 0.8, .0]}
    ]
    
    for case in weather_test_cases:
        try:
            weather_response = requests.post(f"{BASE_URL}/weather-prediction", json=case['data'])
            weather_response.raise_for_status()
            print("Weather Prediction Response:")
            print(json.dumps(weather_response.json(), indent=2))
        except requests.exceptions.RequestException as e:
            print(f"Error during weather prediction request: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON response for weather prediction.")

def test_pest_detection(image_paths):
    """Test the pest detection endpoint with different images"""
    for img_path in image_paths:
        try:
            if not os.path.exists(img_path):
                print(f"Error: Image file not found at {img_path}")
                continue
            
            with open(img_path, "rb") as img_file:
                files = {"image": img_file}
                pest_response = requests.post(f"{BASE_URL}/pest-detection", files=files)
                pest_response.raise_for_status()
                print("Pest Detection Response:")
                print(json.dumps(pest_response.json(), indent=2))
        except requests.exceptions.RequestException as e:
            print(f"Error during pest detection request: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON response for pest detection.")
        except Exception as e:
            print(f"Unexpected error during pest detection: {e}")

def test_irrigation(soil_color, city):
    data = {
        "soil_color": soil_color,
        "city": city
    }
    response = requests.post(f"{BASE_URL}/predict-crop", json=data)
    
    if response.status_code == 200:
        print("Crop Recommendation Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Main execution
if __name__ == "__main__":
        
    print("\nTesting Soil Fertility Classification...")
    test_soil_fertility_classification()

    print("\nTesting Weather Prediction...")
    test_weather_prediction()

    print("\nTesting Pest Detection...")
    test_pest_detection(["E:/KrishiMitra/Scripts/Test_images/Test_snail.jpg"])
        
    print("\nTesting Irrigation Recommendation...")
    test_irrigation("Brown","Pune")