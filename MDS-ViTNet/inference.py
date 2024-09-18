# -*- coding: utf-8 -*-
import os
import time
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from model.Merge_CNN_model import CNNMerge
from model.TranSalNet_ViT_multidecoder import TranSalNet
from utils.visualization import visualization

def init_models(args):
    # Use the path provided by the argument
    path_to_ViT_multidecoder = args.path_to_ViT_multidecoder
    path_to_CNNMerge = args.path_to_CNNMerge

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ViT_multidecoder = TranSalNet()
    model_ViT_multidecoder = model_ViT_multidecoder.to(device)
    model_ViT_multidecoder.load_state_dict(torch.load(path_to_ViT_multidecoder), strict=False)
    model_ViT_multidecoder.eval()

    model_CNNMerge = CNNMerge()
    model_CNNMerge = model_CNNMerge.to(device)
    model_CNNMerge.load_state_dict(torch.load(path_to_CNNMerge))
    model_CNNMerge.eval()

    return model_ViT_multidecoder, model_CNNMerge

def process_image(img_path, model_ViT_multidecoder, model_CNNMerge, device, output_dir, args):
    name_img = os.path.splitext(os.path.basename(img_path))[0]
    img = Image.open(img_path).convert("RGB")
    img = np.array(img) / 255.0
    shape_img = img.shape
    shape_img_w, shape_img_h = shape_img[0], shape_img[1]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = TF.resize(img, (288, 384))
    img = img.type(torch.float32).to(device)  # Ensure img is on the correct device

    # Predict saliency
    with torch.no_grad():  # Ensure gradients are not calculated
        pred_map_1, pred_map_2 = model_ViT_multidecoder(img.unsqueeze(0).to(device))
        pred_map = model_CNNMerge(pred_map_1, pred_map_2)

    # Resize to original image size
    pred_map = TF.resize(pred_map, (shape_img_w, shape_img_h))

    # Convert prediction to grayscale
    pred_map = pred_map.squeeze(0)  # Remove batch dimension
    pred_map = pred_map.squeeze(0)  # Remove channel dimension if it exists
    pred_map = pred_map.cpu().detach().numpy()  # Move to CPU and convert to numpy
    pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min())  # Normalize to [0, 1]
    pred_map = (pred_map * 255).astype(np.uint8)  # Scale to [0, 255]

    # Convert to PIL image and save as grayscale
    pred_map_img = Image.fromarray(pred_map).convert("L")
    output_path = os.path.join(output_dir, f"{name_img}.png")
    pred_map_img.save(output_path)

    print(f"Saved grayscale saliency map to {output_path}")

    # Optionally save color map
    if args.color:
        visualization(
            img_path,
            output_path,
            os.path.join(output_dir, f"{name_img}_color_map.png"),
            max_item=args.max_th,
            min_item=args.min_th
        )

def main(args):
    model_ViT_multidecoder, model_CNNMerge = init_models(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start = time.time()

    img_dir = args.img_path
    output_dir = args.output_dir

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all image files in the directory
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image with tqdm progress bar
    for img_path in tqdm(img_files, desc="Processing images"):
        process_image(img_path, model_ViT_multidecoder, model_CNNMerge, device, output_dir, args)

    print("Total time: ", time.time() - start)
    print("The saliency maps are saved to the specified output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="/path/to/your/images", type=str, help="Path to directory containing images")
    parser.add_argument("--output_dir", default="/path/to/your/results", type=str, help="Path to save output images")
    parser.add_argument("--path_to_ViT_multidecoder", default="./checkpoints/best_model_200_2_0.5_1e-4_40.pth", type=str, help="Path to ViT Multidecoder model")
    parser.add_argument("--path_to_CNNMerge", default="./weights/CNNMerge.pth", type=str, help="Path to CNNMerge model")
    parser.add_argument("--color", default=False, action='store_true', help="Save color map on the image")
    parser.add_argument("--max_th", default=230, type=int, help="Max threshold")
    parser.add_argument("--min_th", default=110, type=int, help="Min threshold")
    args = parser.parse_args()
    main(args)
