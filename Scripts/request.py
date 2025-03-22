import requests
import json
import os
from pathlib import Path
import sys

# Flask base URL
BASE_URL = "http://127.0.0.1:5000"

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

# Main execution
if __name__ == "__main__":

    print("\nTesting Pest Detection...")
    test_pest_detection(["E:/KrishiMitra/Scripts/Test_images/Test_snail.jpg"])