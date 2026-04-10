import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Constants
INPUT_DIR = "data/Pest/Raw_Images"
OUTPUT_DIR = "data/Pest/Processed_Images"
TARGET_SIZE = (224, 224)  # Target image size (width, height)

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Process images
def preprocess_images(input_dir, output_dir, target_size):
    classes = os.listdir(input_dir) # List of class names
    
    for pest_class in classes:
        class_input_path = os.path.join(input_dir, pest_class)
        class_output_path = os.path.join(output_dir, pest_class)
        Path(class_output_path).mkdir(parents=True, exist_ok=True)
        
        for img_name in tqdm(os.listdir(class_input_path), desc=f"Processing {pest_class}"):
            img_path = os.path.join(class_input_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue # Skip if image is not valid
            
            # Resize image
            img = cv2.resize(img, target_size)
            img = img / 255.0  # Normalize to [0,1]

            # Save image
            output_path = os.path.join(class_output_path, img_name)
            cv2.imwrite(output_path, (img * 255).astype(np.uint8))

# Preprocess images
preprocess_images(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)

print("Resized images saved.")
