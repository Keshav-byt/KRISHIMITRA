import requests
import json

# Flask base URL
BASE_URL = "http://127.0.0.1:5000"

# Test Soil Fertility Classification
def test_soil_fertility_classification():
    soil_data = {
        "N": 210,
        "P": 50,
        "K": 300,
        "pH": 6.8,
        "EC": 1.5,
        "OC": 0.9,
        "S": 15,
        "Zn": 1.2,
        "Fe": 5.0,
        "Cu": 1.8,
        "Mn": 1.0,
        "B": 0.5
    }
    try:
        soil_response = requests.post(f"{BASE_URL}/soil-analysis", json=soil_data)
        soil_response.raise_for_status()
        print("Soil Fertility Classification Response:")
        print(soil_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error during soil classification request: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON response for soil classification.")

# Test Weather Prediction
def test_weather_prediction():
    weather_data = [25.0, 0.6, 10.0]
    try:
        weather_response = requests.post(f"{BASE_URL}/weather-prediction", json=weather_data)
        weather_response.raise_for_status()
        print("\nWeather Prediction Response:")
        print(weather_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error during weather prediction request: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON response for weather prediction.")

# Test Pest Detection
def test_pest_detection(image_path):
    try:
        with open(image_path, "rb") as img_file:
            files = {"image": img_file}
            pest_response = requests.post(f"{BASE_URL}/pest-detection", files=files)
            pest_response.raise_for_status()
            print("\nPest Detection Response:")
            print(pest_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error during pest detection request: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON response for pest detection.")
    except FileNotFoundError:
        print("Error: Image file not found.")

# Main execution
if __name__ == "__main__":
    print("Testing Soil Fertility Classification...")
    test_soil_fertility_classification()

    print("\nTesting Weather Prediction...")
    test_weather_prediction()

    print("\nTesting Pest Detection...")
    test_pest_detection("E:\KrishiMitra\Scripts\Test_snail.jpg")
