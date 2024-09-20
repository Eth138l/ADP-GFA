import os
from PIL import Image

# Image folder paths
input_folder = './datasets/AI4VA/val/val_stimuli/'
output_folder = './datasets/AI4VA/val/val_stimuli_resize/'

# Create output folder (if it doesn't exist)
os.makedirs(output_folder, exist_ok=True)

# Target image size
target_size = (512, 512)

# Resize all images in the input folder
def resize_images(input_folder, output_folder, target_size):
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Construct the full input image path
            img_path = os.path.join(input_folder, filename)
            
            # Open the image and resize it
            img = Image.open(img_path)
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
            
            # Construct the output image path
            output_img_path = os.path.join(output_folder, filename)
            
            # Save the resized image
            img_resized.save(output_img_path)
            
            print(f"Processed {filename}")

# Batch process all images
resize_images(input_folder, output_folder, target_size)

print("All images have been resized.")
