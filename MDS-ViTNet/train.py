import os
import time
import copy
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
from IPython.display import clear_output
import bitsandbytes as bnb
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse  # 新增的import

from utils.loss_function import SaliencyLoss
from utils.loss_function import AUC
from utils.data_process import MyDataset, MyTransform
from utils.data_process import MyDatasetCNNMerge, MyTransformCNNMerge
from utils.data_process import preprocess_img, postprocess_img
from utils.data_process import compute_metric, compute_metric_CNNMerge
from utils.data_process import count_parameters

from model.Merge_CNN_model import CNNMerge

# Set flag according to model type
flag = 3  # 0 for TranSalNet_Dense, 1 for TranSalNet_Res, 2 for TranSalNet_ViT, 3 for TranSalNet_ViT_multidecoder

if flag == 0:
    from model.TranSalNet_Dense import TranSalNet
elif flag == 1:
    from model.TranSalNet_Res import TranSalNet
elif flag == 2:
    from model.TranSalNet_ViT import TranSalNet
elif flag == 3:
    from model.TranSalNet_ViT_multidecoder import TranSalNet

def main(p, path_to_save):
    # Paths
    path_images_train = '../data_mds2e/images/train/'
    path_images_val = '../data_mds2e/images/val/'
    path_images_test = '../data_mds2e/images/test/'

    path_maps_train = '../data_mds2e/maps/train/'
    path_maps_val = '../data_mds2e/maps/val/'

    path_train_ids = '../data_mds2e/train_id.csv'
    path_val_ids = '../data_mds2e/val_id.csv'

    # Print the number of files in each dataset folder
    print(f"Number of files in the training image folder '{path_images_train}': {len(os.listdir(path_images_train))}")
    print(f"Number of files in the testing image folder '{path_images_test}': {len(os.listdir(path_images_test))}")
    print(f"Number of files in the validation image folder '{path_images_val}': {len(os.listdir(path_images_val))}")
    print(f"Number of files in the training saliency map folder '{path_maps_train}': {len(os.listdir(path_maps_train))}")
    print(f"Number of files in the validation saliency map folder '{path_maps_val}': {len(os.listdir(path_maps_val))}")

    # Read training and validation ID files
    train_ids = pd.read_excel(path_train_ids)
    val_ids = pd.read_excel(path_val_ids)
    dataset_sizes = {'train': len(train_ids), 'val': len(val_ids)}
    print(dataset_sizes)

    # DataLoader configuration
    batch_size = 2
    shape_r = 288
    shape_c = 384

    train_transform = MyTransform(p=p, shape_r=shape_r, shape_c=shape_c, iftrain=True)
    val_transform = MyTransform(p=p, shape_r=shape_r, shape_c=shape_c, iftrain=False)

    train_set = MyDataset(
        ids=train_ids,
        stimuli_dir=path_images_train,
        saliency_dir=path_maps_train,
        transform=train_transform
    )

    val_set = MyDataset(
        ids=val_ids,
        stimuli_dir=path_images_val,
        saliency_dir=path_maps_val,
        transform=val_transform
    )

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = TranSalNet().to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Training setup
    history_loss_train = []
    history_loss_val = []
    history_loss_train_cc = []
    history_loss_train_sim = []
    history_loss_train_kldiv = []
    history_loss_train_nss = []
    history_loss_train_auc = []
    history_loss_val_cc = []
    history_loss_val_sim = []
    history_loss_val_kldiv = []
    history_loss_val_nss = []
    history_loss_val_auc = []

    # Define lr and step_size
    lr = 1e-4
    step_size = 40

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    loss_fn = SaliencyLoss()

    num_epochs = 200
    best_loss = float('inf')

    # Track for early stopping
    counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for i_batch, sample_batched in tqdm(enumerate(dataloaders[phase])):
                stimuli, smap = sample_batched['image'], sample_batched['saliency']
                stimuli, smap = stimuli.type(torch.float32), smap.type(torch.float32)
                stimuli, smap = stimuli.to(device), smap.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if flag == 3:
                        outputs_1, outputs_2 = model(stimuli)

                        loss_1 = -2 * loss_fn(outputs_1, smap, loss_type='cc')
                        loss_1 -= loss_fn(outputs_1, smap, loss_type='sim')
                        loss_1 += 10 * loss_fn(outputs_1, smap, loss_type='kldiv')

                        loss_2 = -2 * loss_fn(outputs_2, smap, loss_type='cc')
                        loss_2 -= loss_fn(outputs_2, smap, loss_type='sim')
                        loss_2 += 10 * loss_fn(outputs_2, smap, loss_type='kldiv')

                        loss = loss_1 + loss_2
                    else:
                        outputs = model(stimuli)
                        loss = -2 * loss_fn(outputs, smap, loss_type='cc')
                        loss -= loss_fn(outputs, smap, loss_type='sim')
                        loss += 10 * loss_fn(outputs, smap, loss_type='kldiv')

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == 'train':
                    history_loss_train.append(loss.item())
                else:
                    history_loss_val.append(loss.item())

                running_loss += loss.item()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
            elif phase == 'val':
                counter += 1
                if counter > 3:
                    break

        # Save model checkpoint every 40 epochs
        if (epoch + 1) % 40 == 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            checkpoint_path = os.path.join(path_to_save, f'checkpoint_{timestamp}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')

    # Save the best model
    model.load_state_dict(best_model_wts)
    save_path = os.path.join(path_to_save, f'best_model_{num_epochs}_{batch_size}_{p}_{lr}_{step_size}.pth')
    print(f"Saved best model to: {save_path}")
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TranSalNet model')
    parser.add_argument('--p', type=float, default=0.5, help='Probability for MyTransform')
    parser.add_argument('--path_to_save', type=str, default='./checkpoints', help='Path to save model checkpoints')

    args = parser.parse_args()
    main(p=args.p, path_to_save=args.path_to_save)
