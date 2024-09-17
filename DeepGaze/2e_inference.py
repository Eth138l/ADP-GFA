import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
import deepgaze_pytorch

def process_image(image_path, model, device, centerbias_template, output_dir, target_size):
    # 加载图像
    image = Image.open(image_path)

    # 改变图像尺寸
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = np.array(image)

    # 调整中心偏置大小以匹配图像大小
    centerbias = zoom(centerbias_template, (image.shape[0] / centerbias_template.shape[0], image.shape[1] / centerbias_template.shape[1]), order=0, mode='nearest')
    # 重新归一化对数密度
    centerbias -= logsumexp(centerbias)

    # 准备图像和中心偏置张量
    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(device)
    centerbias_tensor = torch.tensor([centerbias]).to(device)

    # 执行显著性预测
    log_density_prediction = model(image_tensor, centerbias_tensor)

    # 转换预测结果为 NumPy 数组
    log_density_prediction_np = log_density_prediction.cpu().detach().numpy()[0, 0]

    # 将预测结果转换为图像并保存到指定目录
    prediction_image = (255 * (log_density_prediction_np - log_density_prediction_np.min()) / (log_density_prediction_np.max() - log_density_prediction_np.min())).astype(np.uint8)
    prediction_image_pil = Image.fromarray(prediction_image)

    # 使用与原始图像相同的文件名保存结果
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    prediction_image_pil.save(output_path)

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Saliency Prediction using DeepGazeIIE.")
    parser.add_argument('--images_path', type=str, default="/home/pod/shared-nvme/AI4VA/data/images/test/", help="Path to the input image directory.")
    parser.add_argument('--results_path', type=str, default="/home/pod/shared-nvme/AI4VA/results/2e_test", help="Path to the output directory where results will be saved.")
    parser.add_argument('--target_size', type=int, nargs=2, default=(1600, 2200), help="Target size for resizing images (width height).")

    args = parser.parse_args()

    # 设置设备为GPU或CPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载预训练的 DeepGazeIIE 模型
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

    # 加载中心偏置数据
    centerbias_template = np.load('centerbias_mit1003.npy')  # 请确保路径正确

    # 确保结果目录存在
    os.makedirs(args.results_path, exist_ok=True)

    # 获取输入目录中的所有图像文件
    image_files = [f for f in os.listdir(args.images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 使用 tqdm 显示进度条
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.images_path, filename)
        process_image(image_path, model, DEVICE, centerbias_template, args.results_path, tuple(args.target_size))

if __name__ == "__main__":
    main()
