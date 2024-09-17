import os
import numpy as np
from PIL import Image
import tqdm


def read_image(image_path):
    """读取显著性图像"""
    return np.array(Image.open(image_path).convert('L'))


def resize_image(image, target_size):
    """调整图像大小"""
    return np.array(Image.fromarray(image).resize((target_size[1], target_size[0]), Image.BILINEAR))


def save_image(image_array, save_path):
    """保存图像，保留输入格式"""
    img = Image.fromarray(image_array)
    img.save(save_path)


def merge_predictions(prediction_folders, output_folder, weights):
    """
    加权合并多个路径中的显著性图像，图像大小调整为第一个路径的图像大小
    prediction_folders: 包含显著性图像的文件夹列表
    output_folder: 融合结果保存的文件夹
    weights: 每个文件夹对应的权重列表
    """
    if len(prediction_folders) != len(weights):
        raise ValueError("预测文件夹的数量必须与权重列表的长度一致")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取每个文件夹中的文件名列表（假设所有文件夹中图像数量相同）
    prediction_files = os.listdir(prediction_folders[0])

    for file_name in tqdm.tqdm(prediction_files, desc="Merging predictions"):
        # 跳过以 `.` 开头的文件或文件夹
        if file_name.startswith('.'):
            continue

        # 读取第一个路径的图像，作为目标大小的基准
        base_image = read_image(os.path.join(prediction_folders[0], file_name))
        base_size = base_image.shape[:2]  # (height, width)

        # 初始化融合后的图像，和第一个路径图像大小相同
        merged_image = np.zeros_like(base_image, dtype=np.float32)

        # 遍历每个路径并进行图像融合
        for folder, weight in zip(prediction_folders, weights):
            image_path = os.path.join(folder, file_name)
            image = read_image(image_path)

            # 如果当前图像大小与第一个路径的图像不一致，则调整大小
            if image.shape[:2] != base_size:
                image = resize_image(image, target_size=base_size)

            # 累加当前图像乘以权重
            merged_image += weight * image.astype(np.float32)

        # 保存结果到输出文件夹，保留原始文件格式
        save_path = os.path.join(output_folder, file_name)
        save_image(merged_image.astype(np.uint8), save_path)


# 使用示例
prediction_folders = ['/home/pod/shared-nvme/AI4VA/results/MDS_test_trained200_2_1e-4_40', '/home/pod/shared-nvme/AI4VA/results/MDS_test_trained200_0.6_2_1e-4_40', '/home/pod/shared-nvme/AI4VA/results/SUM_newK5_heibai_gaosi_31', '/home/pod/shared-nvme/AI4VA/results/2e_test']  # 显著性图像的文件夹列表
output_folder = '/home/pod/shared-nvme/AI4VA/results/2MDS_mixSUMk5IIE'  # 融合结果保存的文件夹
weights = [0.35, 0.35, 0.15, 0.15]  # 每个路径对应的权重，确保总和为1

merge_predictions(prediction_folders, output_folder, weights)