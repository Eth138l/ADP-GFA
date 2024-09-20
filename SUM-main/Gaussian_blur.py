import cv2
import os

# Input and output folder paths
input_folder = './SUM_newK5_heibai/'  # Replace with your input folder path
output_folder = '../results/SUM_newK5_gaosi/'  # Replace with your output folder path

# Check if the output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set the kernel size for Gaussian blur, the larger the value, the more blur; must be odd
kernel_size = (27, 27)

# Iterate through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):  # Only process specified image formats
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale mode

        # Apply Gaussian blur to the image
        blurred_img = cv2.GaussianBlur(img, kernel_size, 0)

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, blurred_img)

print("Processing complete. Blurred images have been saved to the output folder.")
