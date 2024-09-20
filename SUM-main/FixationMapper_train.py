import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_fixation_map(saliency_map, threshold=0.15, num_fixations=100, point_size=3):
    """
    Generate a fixation map from a saliency map, using larger points to represent fixation points and ensuring coverage of salient regions.

    Parameters:
    - saliency_map: Input saliency map, a grayscale image with values between 0-255.
    - threshold: Threshold to limit the area for generating fixation points, the value should be between 0-1.
    - num_fixations: Number of fixation points to generate.
    - point_size: Size of the fixation points for visualization.

    Returns:
    - fixation_map: A fixation map where fixation points are marked as 1 and the background as 0.
    """
    # Normalize the saliency map to [0, 1]
    saliency_map = saliency_map.astype(np.float32) / 255.0

    # Threshold processing to select areas with saliency above the threshold
    high_saliency_indices = np.where(saliency_map >= threshold)

    # If no areas meet the threshold, return a blank fixation map
    if len(high_saliency_indices[0]) == 0:
        return np.zeros_like(saliency_map)

    # Create a blank fixation map
    fixation_map = np.zeros_like(saliency_map)

    # Ensure coverage of salient areas: generate points in highly salient areas
    for i in range(len(high_saliency_indices[0])):
        y, x = high_saliency_indices[0][i], high_saliency_indices[1][i]
        # Expand the selected point into a small region, e.g., 3x3 or 5x5
        fixation_map[max(0, y-point_size//2):min(fixation_map.shape[0], y+point_size//2+1),
                     max(0, x-point_size//2):min(fixation_map.shape[1], x+point_size//2+1)] = 1

    # Optionally, randomly select some points from high saliency areas as primary fixation points
    if len(high_saliency_indices[0]) > num_fixations:
        selected_indices = np.random.choice(len(high_saliency_indices[0]), num_fixations, replace=False)
        for idx in selected_indices:
            y, x = high_saliency_indices[0][idx], high_saliency_indices[1][idx]
            # Increase the size of the primary fixation points for better representation
            fixation_map[max(0, y-point_size//2):min(fixation_map.shape[0], y+point_size//2+1),
                         max(0, x-point_size//2):min(fixation_map.shape[1], x+point_size//2+1)] = 1

    return fixation_map

def process_images(input_folder, output_folder, threshold=0.15, num_fixations=100, point_size=2):
    """
    Read all image files from the input folder, generate fixation maps, and save them to the output folder.

    Parameters:
    - input_folder: Path to the input image folder.
    - output_folder: Path to the output image folder.
    - threshold: Saliency threshold for generating fixation maps.
    - num_fixations: Number of fixation points to generate for each image.
    - point_size: Size of the fixation points.
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Construct the full file path
        input_path = os.path.join(input_folder, filename)
        
        # Ensure only image files are processed
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Read the saliency map (grayscale image)
        saliency_map = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Generate the fixation map
        fixation_map = generate_fixation_map(saliency_map, threshold=threshold, num_fixations=num_fixations, point_size=point_size)

        # Construct the output file path
        output_path = os.path.join(output_folder, filename)

        # Save the fixation map
        cv2.imwrite(output_path, fixation_map * 255)  # Multiply by 255 to save as an 8-bit image

        # Optional: Display processing progress
        print(f'Processed and saved: {output_path}')

# Define folder paths
input_folder = 'datasets/AI4VA/train/train_saliency'
output_folder = 'datasets/AI4VA/train/train_fixation'

# Process all images
process_images(input_folder, output_folder, threshold=0.15, num_fixations=100, point_size=2)
