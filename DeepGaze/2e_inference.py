import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm  # Import tqdm for displaying progress bar
import deepgaze_pytorch

def process_image(image_path, model, device, centerbias_template, output_dir, target_size):
    # Load the image
    image = Image.open(image_path)

    # Resize the image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = np.array(image)

    # Adjust center bias size to match image size
    centerbias = zoom(centerbias_template, (image.shape[0] / centerbias_template.shape[0], image.shape[1] / centerbias_template.shape[1]), order=0, mode='nearest')
    # Re-normalize log density
    centerbias -= logsumexp(centerbias)

    # Prepare image and center bias tensors
    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(device)
    centerbias_tensor = torch.tensor([centerbias]).to(device)

    # Perform saliency prediction
    log_density_prediction = model(image_tensor, centerbias_tensor)

    # Convert prediction results to NumPy array
    log_density_prediction_np = log_density_prediction.cpu().detach().numpy()[0, 0]

    # Convert prediction results to image and save to the specified directory
    prediction_image = (255 * (log_density_prediction_np - log_density_prediction_np.min()) / (log_density_prediction_np.max() - log_density_prediction_np.min())).astype(np.uint8)
    prediction_image_pil = Image.fromarray(prediction_image)

    # Save results with the same filename as the original image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    prediction_image_pil.save(output_path)

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Saliency Prediction using DeepGazeIIE.")
    parser.add_argument('--images_path', type=str, default="../data_mds2e/images/test/", help="Path to the input image directory.")
    parser.add_argument('--results_path', type=str, default="../results/2e_test_1622", help="Path to the output directory where results will be saved.")
    parser.add_argument('--target_size', type=int, nargs=2, default=(1600, 2200), help="Target size for resizing images (width height).")

    args = parser.parse_args()

    # Set device to GPU or CPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained DeepGazeIIE model
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

    # Load center bias data
    centerbias_template = np.load('centerbias_mit1003.npy')

    # Ensure the results directory exists
    os.makedirs(args.results_path, exist_ok=True)

    # Get all image files from the input directory
    image_files = [f for f in os.listdir(args.images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Use tqdm to display progress bar
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.images_path, filename)
        process_image(image_path, model, DEVICE, centerbias_template, args.results_path, tuple(args.target_size))

if __name__ == "__main__":
    main()
