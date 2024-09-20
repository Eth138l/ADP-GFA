import cv2
import numpy as np
import os

def convert_heatmap_to_bw(heatmap_path, output_path):
    # Read heatmap image
    heatmap = cv2.imread(heatmap_path)

    # Check if the image was successfully loaded
    if heatmap is None:
        print(f"Failed to load image {heatmap_path}")
        return

    # Convert to grayscale image
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    # Normalize to [0, 255] and convert to unsigned 8-bit integer
    normalized_gray = cv2.normalize(gray_heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Save black and white saliency image
    cv2.imwrite(output_path, normalized_gray)

    print(f"Converted {heatmap_path} to black and white and saved as {output_path}")

# Example usage
input_folder = './SUM_newK5/'  # Heatmap folder
output_folder = './SUM_newK5_heibai/'  # Output folder

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Modify based on image format
        heatmap_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        convert_heatmap_to_bw(heatmap_path, output_path)
