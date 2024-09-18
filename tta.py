import os
import numpy as np
from PIL import Image
import tqdm
import argparse


def read_image(image_path):
    """Read the saliency image"""
    return np.array(Image.open(image_path).convert('L'))


def resize_image(image, target_size):
    """Resize the image"""
    return np.array(Image.fromarray(image).resize((target_size[1], target_size[0]), Image.BILINEAR))


def save_image(image_array, save_path):
    """Save the image while preserving input format"""
    img = Image.fromarray(image_array)
    img.save(save_path)


def merge_predictions(prediction_folders, output_folder, weights):
    """
    Weighted merging of saliency images from multiple directories.
    Image size is adjusted to match the first directory's image size.

    prediction_folders: List of directories containing saliency images.
    output_folder: Directory where the merged results will be saved.
    weights: List of weights corresponding to each directory.
    """
    if len(prediction_folders) != len(weights):
        raise ValueError("The number of prediction folders must match the length of the weights list")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of filenames from the first folder (assuming all folders contain the same number of images)
    prediction_files = os.listdir(prediction_folders[0])

    for file_name in tqdm.tqdm(prediction_files, desc="Merging predictions"):
        # Skip files or folders starting with `.`
        if file_name.startswith('.'):
            continue

        # Read the first directory's image as the base for the target size
        base_image = read_image(os.path.join(prediction_folders[0], file_name))
        base_size = base_image.shape[:2]  # (height, width)

        # Initialize the merged image, same size as the first folder's image
        merged_image = np.zeros_like(base_image, dtype=np.float32)

        # Iterate through each directory and merge the images
        for folder, weight in zip(prediction_folders, weights):
            image_path = os.path.join(folder, file_name)
            image = read_image(image_path)

            # Resize the image if its size doesn't match the base size
            if image.shape[:2] != base_size:
                image = resize_image(image, target_size=base_size)

            # Accumulate the current image multiplied by the weight
            merged_image += weight * image.astype(np.float32)

        # Save the result to the output folder, preserving the original file format
        save_path = os.path.join(output_folder, file_name)
        save_image(merged_image.astype(np.uint8), save_path)


def main():
    parser = argparse.ArgumentParser(description="Merge saliency predictions with weighted averaging.")
    parser.add_argument(
        '--prediction_folders',
        nargs='+',
        default=[
            './results/MDS_test_trained200_2_0.5_1e-4_40',
            './results/MDS_test_trained200_2_0.6_1e-4_40',
            './results/SUM_newK5_gaosi',
            './results/2e_test_1622'
        ],
        help='List of directories containing saliency images'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='./results/2MDS_mixSUMk5IIE',
        help='Directory to save merged results'
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        default=[0.4, 0.45, 0.1, 0.05],
        help='Weights corresponding to each prediction folder'
    )

    args = parser.parse_args()

    if len(args.prediction_folders) != len(args.weights):
        raise ValueError("The number of prediction folders must match the number of weights")

    # Run the merge predictions function
    merge_predictions(args.prediction_folders, args.output_folder, args.weights)


if __name__ == '__main__':
    main()
